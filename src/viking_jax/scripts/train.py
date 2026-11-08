#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a Viking model against a CARFAC-domain DAL objective.

Mirrors `seanet/scripts/train.py` while exposing `VikingGenerator` filterbank,
minimum-phase FIR, parallel-LSTM/hub, and hearing-aid gain flags.
"""

import dataclasses
import math
import os
import typing
from collections.abc import Sequence
from datetime import datetime

import jax
import optax
from absl import app
from absl import flags
from absl import logging
from etils import epath
from hp_acoustic import audio_constants
from hp_acoustic.data import pipeline
from hp_acoustic.data import transforms
from hp_acoustic.data.sources import color_noise
from hp_acoustic.data.sources import processed
from hp_acoustic.evaluation import evaluator
from hp_acoustic.losses import representation_losses
from hp_acoustic.losses import terms as terms_lib
from hp_acoustic.losses import waveform_losses
from hp_acoustic.models import carfac
from hp_acoustic.models import cochlea
from hp_acoustic.models import context as auditory_context
from hp_acoustic.training import callbacks as callbacks_lib
from hp_acoustic.training import checkpointing
from hp_acoustic.training import data as data_lib
from hp_acoustic.training import loop

from viking_jax import audio_model as _viking_audio_model  # noqa: F401
from viking_jax import viking_jax
from viking_jax.scripts import flag_utils

jax.config.update('jax_debug_nans', False)

_INPUT_BASE_DIR = flags.DEFINE_string(
  'input_base_dir',
  None,
  'Directory of a corpus prepared by prepare_librispeech.py, holding '
  'train/ and test/ subdirectories of clip shards.',
  required=True,
)
_OUTPUT_BASE_DIR = flags.DEFINE_string(
  'output_base_dir',
  None,
  'Path to the base directory for saving runs, models, and reports.',
  required=True,
)
_RUN_NAME = flags.DEFINE_string(
  'run_name',
  None,
  'Optional run name. Defaults to viking_<timestamp> if not set.',
)
_MODEL_CHECKPOINT_PATH = flags.DEFINE_string(
  'model_checkpoint_path',
  None,
  'Path to a specific model checkpoint to resume from.',
)
_MAX_STEPS = flags.DEFINE_integer(
  'max_steps',
  None,
  'Ceiling on optimizer steps. None or <= 0 for no limit.',
)
_NUM_VAL_EXAMPLES = flags.DEFINE_integer(
  'num_val_examples',
  None,
  'Number of validation examples to score. None or <= 0 for all.',
)
_NUM_ANALYSIS_CLIPS = flags.DEFINE_integer(
  'num_analysis_clips',
  50,
  'Number of test clips to score in the final report. None or <= 0 for all.',
)
_GLOBAL_BATCH_SIZE = flags.DEFINE_integer(
  'global_batch_size', 32, 'Global batch size for training.'
)
_TRAINING_EXAMPLE_SECONDS = flags.DEFINE_float(
  'training_example_seconds', 0.512, 'Duration of audio training examples.'
)
_MIN_ACTIVE_FRACTION = flags.DEFINE_float(
  'min_active_fraction',
  0.0,
  'Drops training chunks whose detected speech span covers less than this '
  'fraction of the chunk. 0 disables filtering.',
  lower_bound=0.0,
  upper_bound=1.0,
)
_TARGET_SAMPLE_RATE = flags.DEFINE_integer(
  'target_sample_rate',
  audio_constants.DEFAULT_SAMPLE_RATE,
  'Training sample rate in Hz. Saved in checkpoints.',
  lower_bound=1,
)
_MAX_EPOCHS = flags.DEFINE_integer(
  'max_epochs', 20, 'Ceiling on complete passes over the training data.'
)
_CHECKPOINT_STEP_FREQ = flags.DEFINE_integer(
  'checkpoint_step_freq',
  25,
  'Run a validation pass and write a checkpoint every this many steps.',
)
_EARLY_STOPPING_PATIENCE_CHECKS = flags.DEFINE_integer(
  'early_stopping_patience_checks',
  0,
  'Stop after this many consecutive validation passes without improvement.',
)
_EARLY_STOPPING_MIN_DELTA = flags.DEFINE_float(
  'early_stopping_min_delta',
  0.0,
  'Minimum decrease in weighted validation loss to count as improvement.',
)

# Optimizer & seed flags
_LEARNING_RATE = flags.DEFINE_float(
  'learning_rate', 0.0005, 'Learning rate for the Adam optimizer.'
)
_BETA1 = flags.DEFINE_float('beta1', 0.5, 'Adam optimizer beta1.')
_BETA2 = flags.DEFINE_float('beta2', 0.9, 'Adam optimizer beta2.')
_SEED = flags.DEFINE_integer('seed', 0, 'Model parameter initialization seed.')
_DATA_SEED = flags.DEFINE_integer('data_seed', 42, 'Dataset shuffle seed.')

# Loss flags (matching seanet/scripts/train.py)
_LAMBDA_SSIM = flags.DEFINE_float('lambda_ssim', 1.0, 'Weight for SAI SSIM.')
_LAMBDA_PND_SAI = flags.DEFINE_float(
  'lambda_pnd_sai', 1.0, 'Weight for SAI PND.'
)
_LAMBDA_DRIFT = flags.DEFINE_float(
  'lambda_drift',
  terms_lib.DEFAULT_DRIFT_WEIGHT,
  'Weight for low-frequency drift loss.',
)
_LAMBDA_L1_WAVEFORM = flags.DEFINE_float(
  'lambda_l1_waveform',
  terms_lib.DEFAULT_L1_WAVEFORM_WEIGHT,
  'Weight for L1 waveform level-anchor loss.',
)
_LAMBDA_L1_FFT = flags.DEFINE_float('lambda_l1_fft', 0.0, 'Weight for L1 FFT.')
_LAMBDA_STFT = flags.DEFINE_float('lambda_stft', 0.0, 'Weight for MR-STFT.')
_LAMBDA_EUCLIDEAN = flags.DEFINE_float(
  'lambda_euclidean', 0.0, 'Weight for SAI MSE.'
)
_LAMBDA_L1_NAP = flags.DEFINE_float(
  'lambda_l1_nap', 0.0, 'Weight for pointwise NAP L1.'
)

_PND_SAI_ALPHA = flags.DEFINE_float(
  'pnd_sai_alpha', 0.5, 'Alpha for Partial Normalized Distance on SAI.'
)
_PND_SAI_BETA = flags.DEFINE_float(
  'pnd_sai_beta', 0.8, 'Beta for Partial Normalized Distance on SAI.'
)
_IMPAIRED_OHC_HEALTH = flags.DEFINE_float(
  'impaired_ohc_health',
  0.5,
  'OHC health for the impaired CARFAC model (0.0 to 1.0).',
)
_SSIM_TIMEWINSZ = flags.DEFINE_integer(
  'ssim_timewinsz', 36, 'SSIM window length in time.'
)
_SSIM_CHANWINSZ = flags.DEFINE_integer(
  'ssim_chanwinsz', 6, 'SSIM window length in channels.'
)
_SSIM_DYNAMIC_RANGE = flags.DEFINE_float(
  'ssim_dynamic_range',
  representation_losses.DEFAULT_SSIM_DYNAMIC_RANGE,
  'Peak SAI value the SSIM stabilisers scale against.',
)

# Data augmentation flags
_CLEAN_CLEAN_LOOP = flags.DEFINE_boolean(
  'clean_clean_loop',
  False,
  'Feed clean target as input to measure the CARFAC/SAI loss floor.',
)
_NOISE_SNR_DB_LOW = flags.DEFINE_float(
  'noise_snr_db_low', -5.0, 'Low bound of per-example SNR in dB.'
)
_NOISE_SNR_DB_HIGH = flags.DEFINE_float(
  'noise_snr_db_high', 10.0, 'High bound of per-example SNR in dB.'
)
_LEVEL_DB_SPL_LOW = flags.DEFINE_float(
  'level_db_spl_low', 50.0, 'Low bound of presentation level in dB SPL.'
)
_LEVEL_DB_SPL_HIGH = flags.DEFINE_float(
  'level_db_spl_high', 80.0, 'High bound of presentation level in dB SPL.'
)
_LEVEL_REFERENCE = flags.DEFINE_enum(
  'level_reference',
  'long_term',
  ['long_term', 'active'],
  'Whether level/SNR are measured over whole chunk or active speech span.',
)
_RAMP_SECONDS = flags.DEFINE_float(
  'ramp_seconds',
  0.005,
  'Cosine-squared edge fade duration in seconds (0 disables).',
  lower_bound=0.0,
)

# Viking architecture flags
_USE_LOW_LATENCY_FILTERBANKS = flags.DEFINE_boolean(
  'use_low_latency_filterbanks',
  True,
  'Use ultra-low-latency sinc analysis + FIR synthesis filterbanks.',
)
_CONVERT_TO_MIN_PHASE = flags.DEFINE_boolean(
  'convert_to_min_phase',
  True,
  'Convert synthesis FIR taps to minimum phase via real cepstrum.',
)
_FRAME_LENGTH = flags.DEFINE_integer(
  'frame_length', 256, 'STFT frame length when low-latency mode is False.'
)
_FRAME_STEP = flags.DEFINE_integer(
  'frame_step', 16, 'STFT hop size when low-latency mode is False.'
)
_SHIFT_FACTOR = flags.DEFINE_integer(
  'shift_factor',
  64,
  'Hop size in samples in low-latency mode (64 samples = 2 ms at 32 kHz).',
)
_NUM_SUBBANDS = flags.DEFINE_integer(
  'num_subbands', 512, 'Number of subbands in low-latency mode.'
)
_PROTOTYPE_FILTER_LENGTH = flags.DEFINE_integer(
  'prototype_filter_length', 512, 'Prototype sinc filter length.'
)
_NUM_TAPS = flags.DEFINE_integer(
  'num_taps', 256, 'Synthesis FIR filter tap count in low-latency mode.'
)
_NUM_PARALLEL = flags.DEFINE_integer(
  'num_parallel', 4, 'Number of parallel frequency-band LSTM branches.'
)
_LSTM_DEPTH = flags.DEFINE_integer(
  'lstm_depth', 2, 'Number of stacked recurrent LSTM layers.'
)
_LSTM_UNITS = flags.DEFINE_integer(
  'lstm_units', 32, 'Hidden units per parallel LSTM branch.'
)
_HUB_UNITS = flags.DEFINE_integer(
  'hub_units', 8, 'Units in cross-band dense hub projections.'
)
_IN_MAGNITUDE_COMPRESSION = flags.DEFINE_float(
  'in_magnitude_compression', 0.3, 'Power-law input magnitude compression.'
)
_MAX_MASK_GAIN = flags.DEFINE_float(
  'max_mask_gain',
  20.0,
  'Maximum subband insertion gain (> 1.0 enables hearing-aid amplification).',
)
_INIT_MASK_GAIN = flags.DEFINE_float(
  'init_mask_gain',
  1.0,
  'Initial subband insertion gain (e.g. 1.0 for unity, > 1.0 for audiometric warm-start).',
)


@flags.validator(
  'training_example_seconds',
  message='must be a positive, finite number of seconds',
)
def _example_seconds_is_positive(seconds: float) -> bool:
  return math.isfinite(seconds) and seconds > 0


def _clean_target(example, rng):
  del rng
  return dataclasses.replace(example, noisy=example.clean)


def build_viking_model_from_flags() -> viking_jax.VikingModel:
  """Constructs `VikingModel` from CLI flags."""
  generator = viking_jax.VikingGenerator(
    frame_length=_FRAME_LENGTH.value,
    frame_step=_FRAME_STEP.value,
    use_low_latency_filterbanks=_USE_LOW_LATENCY_FILTERBANKS.value,
    shift_factor=_SHIFT_FACTOR.value,
    num_subbands=_NUM_SUBBANDS.value,
    prototype_filter_length=_PROTOTYPE_FILTER_LENGTH.value,
    convert_to_min_phase=_CONVERT_TO_MIN_PHASE.value,
    num_taps=_NUM_TAPS.value,
    num_parallel=_NUM_PARALLEL.value,
    lstm_depth=_LSTM_DEPTH.value,
    lstm_units=_LSTM_UNITS.value,
    hub_units=_HUB_UNITS.value,
    in_magnitude_compression=_IN_MAGNITUDE_COMPRESSION.value,
    max_mask_gain=_MAX_MASK_GAIN.value,
    init_mask_gain=_INIT_MASK_GAIN.value,
  )
  return viking_jax.VikingModel(module=generator)


def _build_terms(sample_rate: int) -> tuple[terms_lib.LossTerm, ...]:
  """Constructs the active `LossTerm` instances specified by flags."""
  terms: list[terms_lib.LossTerm] = []
  if _LAMBDA_SSIM.value > 0:
    terms.append(
      representation_losses.SaiSsimLoss(
        weight=_LAMBDA_SSIM.value,
        timewinsz=_SSIM_TIMEWINSZ.value,
        chanwinsz=_SSIM_CHANWINSZ.value,
        dynamic_range=_SSIM_DYNAMIC_RANGE.value,
      )
    )
  if _LAMBDA_PND_SAI.value > 0:
    terms.append(
      representation_losses.SaiPndLoss(
        weight=_LAMBDA_PND_SAI.value,
        alpha=_PND_SAI_ALPHA.value,
        beta=_PND_SAI_BETA.value,
      )
    )
  if _LAMBDA_DRIFT.value > 0:
    terms.append(
      waveform_losses.LowFrequencyDriftLoss(
        weight=_LAMBDA_DRIFT.value, sample_rate=sample_rate
      )
    )
  if _LAMBDA_L1_WAVEFORM.value > 0:
    terms.append(
      waveform_losses.L1WaveformLoss(weight=_LAMBDA_L1_WAVEFORM.value)
    )
  if _LAMBDA_L1_FFT.value > 0:
    terms.append(waveform_losses.L1FftLoss(weight=_LAMBDA_L1_FFT.value))
  if _LAMBDA_STFT.value > 0:
    terms.append(
      waveform_losses.MultiresolutionStftLoss(
        weight=_LAMBDA_STFT.value, sample_rate=sample_rate
      )
    )
  if _LAMBDA_EUCLIDEAN.value > 0:
    terms.append(
      representation_losses.SaiMseLoss(weight=_LAMBDA_EUCLIDEAN.value)
    )
  if _LAMBDA_L1_NAP.value > 0:
    terms.append(representation_losses.NapL1Loss(weight=_LAMBDA_L1_NAP.value))
  return tuple(terms)


def _build_optimizer() -> optax.GradientTransformation:
  return optax.adam(
    learning_rate=_LEARNING_RATE.value,
    b1=_BETA1.value,
    b2=_BETA2.value,
  )


def _build_transforms(sample_rate: int) -> tuple[transforms.Transform, ...]:
  fade: tuple[transforms.Transform, ...] = ()
  if _RAMP_SECONDS.value > 0:
    fade = (transforms.CosineWindow(ramp_seconds=_RAMP_SECONDS.value),)

  chain: list[transforms.Transform] = []
  if _CLEAN_CLEAN_LOOP.value:
    chain.append(_clean_target)
  elif _NOISE_SNR_DB_HIGH.value >= _NOISE_SNR_DB_LOW.value:
    chain.append(
      transforms.BlendSource(
        sources=(
          color_noise.ColorNoise(color='white', sample_rate=sample_rate),
          color_noise.ColorNoise(color='brown', sample_rate=sample_rate),
        ),
        snr_db=(_NOISE_SNR_DB_LOW.value, _NOISE_SNR_DB_HIGH.value),
        num_sources=(1, 1),
        snr_reference=typing.cast(
          transforms.LevelReference, _LEVEL_REFERENCE.value
        ),
      )
    )

  if _LEVEL_DB_SPL_HIGH.value >= _LEVEL_DB_SPL_LOW.value:
    chain.append(
      transforms.NormalizeLevel(
        level_db_spl=(_LEVEL_DB_SPL_LOW.value, _LEVEL_DB_SPL_HIGH.value),
        level_reference=typing.cast(
          transforms.LevelReference, _LEVEL_REFERENCE.value
        ),
      )
    )

  chain.extend(fade)
  return tuple(chain)


def _build_data(sample_rate: int) -> pipeline.DataConfig:
  return pipeline.DataConfig(
    sources=(processed.ProcessedClips(_INPUT_BASE_DIR.value),),
    transforms=_build_transforms(sample_rate),
    global_batch_size=_GLOBAL_BATCH_SIZE.value,
    sample_rate=sample_rate,
    chunk_seconds=_TRAINING_EXAMPLE_SECONDS.value,
    min_active_fraction=_MIN_ACTIVE_FRACTION.value or None,
    max_epochs=_MAX_EPOCHS.value,
    seed=_DATA_SEED.value,
  )


def main(argv: Sequence[str]) -> None:
  del argv

  run_name = (
    _RUN_NAME.value or f"viking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  )
  sample_rate = _TARGET_SAMPLE_RATE.value
  model = build_viking_model_from_flags()
  data_config = _build_data(sample_rate)
  terms = _build_terms(sample_rate)
  optimizer = _build_optimizer()

  hop = (
    model.module.shift_factor
    if model.module.use_low_latency_filterbanks
    else model.module.frame_step
  )
  logging.info(
    'Training Viking at %d Hz: %d-sample examples (%.3f s), hop=%d (%.2f ms)',
    sample_rate,
    data_config.chunk_samples,
    data_config.chunk_seconds,
    hop,
    1000.0 * hop / sample_rate,
  )

  cochleae = cochlea.CochleaPair(
    healthy=carfac.build(sample_rate),
    impaired=carfac.build(sample_rate, ohc_health=_IMPAIRED_OHC_HEALTH.value),
  )

  run_config = checkpointing.RunConfig(
    target_sample_rate=sample_rate,
    impaired_ohc_health=_IMPAIRED_OHC_HEALTH.value,
    model=model.get_config(),
  )

  callbacks = [
    callbacks_lib.Checkpoint(
      output_dir=_OUTPUT_BASE_DIR.value,
      every=_CHECKPOINT_STEP_FREQ.value,
      run_name=run_name,
      run_config=run_config,
    ),
    callbacks_lib.PlotLossCurves(
      output_dir=_OUTPUT_BASE_DIR.value,
      run_name=run_name,
    ),
    callbacks_lib.SaveValidationMetrics(
      output_dir=_OUTPUT_BASE_DIR.value,
      run_name=run_name,
    ),
  ]

  num_analysis_clips = flag_utils.positive_or_none(_NUM_ANALYSIS_CLIPS.value)
  analysis = None
  if num_analysis_clips is not None:
    analysis = data_lib.analysis_samples(
      data_config, num_analysis_clips=num_analysis_clips
    )
  if analysis and not evaluator.hearing_loss_supported(
    _IMPAIRED_OHC_HEALTH.value
  ):
    logging.warning(
      'Skipping final evaluation: HASPI/HASQI currently support only flat '
      'OHC health 0.5, got %s.',
      _IMPAIRED_OHC_HEALTH.value,
    )

  init_params = None
  init_opt_state = None
  initial_step = 0
  if _MODEL_CHECKPOINT_PATH.value:
    checkpointing.require_checkpoint(_MODEL_CHECKPOINT_PATH.value)
    init_params, init_opt_state = checkpointing.restore_model(
      _MODEL_CHECKPOINT_PATH.value, model, optimizer
    )
    restored_step = checkpointing.step_from_path(_MODEL_CHECKPOINT_PATH.value)
    if restored_step is not None:
      initial_step = restored_step

  should_stop = None
  if _EARLY_STOPPING_PATIENCE_CHECKS.value > 0:
    should_stop = loop.patience_stopper(
      _EARLY_STOPPING_PATIENCE_CHECKS.value,
      _EARLY_STOPPING_MIN_DELTA.value,
    )

  result = loop.fit(
    model=model,
    data=pipeline.build(data_config),
    loss=terms,
    context=auditory_context.build(cochleae),
    optimizer=optimizer,
    max_steps=flag_utils.positive_or_none(_MAX_STEPS.value),
    eval_every=_CHECKPOINT_STEP_FREQ.value,
    val_data=data_lib.validation_batches(
      data_config,
      num_val_examples=flag_utils.positive_or_none(_NUM_VAL_EXAMPLES.value),
    ),
    init_params=init_params,
    init_opt_state=init_opt_state,
    initial_step=initial_step,
    seed=_SEED.value,
    callbacks=callbacks,
    should_stop=should_stop,
  )

  if (
    analysis
    and analysis.samples
    and evaluator.hearing_loss_supported(_IMPAIRED_OHC_HEALTH.value)
  ):
    report_directory = os.path.join(
      _OUTPUT_BASE_DIR.value, run_name, 'testing_report'
    )
    epath.Path(report_directory).mkdir(parents=True, exist_ok=True)
    evaluator.run_paper_evaluation(
      final_params_g=result.params,
      generator_model=model,
      run_name=os.path.join(_OUTPUT_BASE_DIR.value, run_name),
      report_directory=report_directory,
      num_analysis_clips=len(analysis.samples),
      t_sr=analysis.sample_rate,
      impaired_ohc_health=_IMPAIRED_OHC_HEALTH.value,
      test_data=analysis.samples,
    )


if __name__ == '__main__':
  app.run(main)
