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

"""Evaluates a saved Viking, SHACA, or SEANet DAL checkpoint on a corpus."""

import math
import typing
from collections.abc import Sequence

import jax
from absl import app
from absl import flags
from absl import logging
from hp_acoustic import audio_constants
from hp_acoustic.data import transforms
from hp_acoustic.evaluation import report
from hp_acoustic.models import audio_model
from hp_acoustic.training import checkpointing

from viking_jax import audio_model as _viking_audio_model  # noqa: F401
from viking_jax import viking_jax
from viking_jax.scripts import flag_utils

try:
  from seanet.models import audio_model as _seanet_audio_model  # noqa: F401
except ImportError:
  pass

try:
  from shaca_jax import audio_model as _shaca_audio_model  # noqa: F401
except ImportError:
  pass

_MODEL_CHECKPOINT_PATH = flags.DEFINE_string(
  'model_checkpoint_path',
  None,
  'Path to the saved checkpoint directory to evaluate.',
  required=True,
)
_INPUT_BASE_DIR = flags.DEFINE_string(
  'input_base_dir',
  None,
  'Directory of a corpus prepared by prepare_librispeech.py.',
  required=True,
)
_REPORT_DIRECTORY = flags.DEFINE_string(
  'report_directory',
  None,
  'Directory to write full_metrics_report.csv, denoised audio, and charts.',
  required=True,
)
_NUM_ANALYSIS_CLIPS = flags.DEFINE_integer(
  'num_analysis_clips',
  50,
  'Number of test clips to score (None or <= 0 for all).',
)
_RUN_NAME = flags.DEFINE_string(
  'run_name',
  'viking_evaluation',
  'Name used to label evaluation charts.',
)
_TARGET_SAMPLE_RATE = flags.DEFINE_integer(
  'target_sample_rate',
  audio_constants.DEFAULT_SAMPLE_RATE,
  'Sample rate of test audio (normally read from checkpoint config.json).',
)
_IMPAIRED_OHC_HEALTH = flags.DEFINE_float(
  'impaired_ohc_health',
  0.5,
  'OHC health of the impaired CARFAC model (0.0 to 1.0).',
)
_NUM_NAP_HEATMAPS = flags.DEFINE_integer(
  'num_nap_heatmaps',
  5,
  'Number of clips to render NAP heatmaps for (0 skips).',
)
_RESTORE_ON_CPU = flags.DEFINE_bool(
  'restore_on_cpu',
  False,
  'Force checkpoint restore onto host CPU.',
)
_LEVEL_REFERENCE = flags.DEFINE_enum(
  'level_reference',
  'long_term',
  ['long_term', 'active'],
  'How test audio level and SNR are measured.',
)
_NOISE_SEED = flags.DEFINE_integer(
  'noise_seed',
  0,
  'Fixed seed for synthesized noise so models are evaluated on identical audio.',
)


def main(argv: Sequence[str]) -> int | None:
  del argv
  checkpoint_path = _MODEL_CHECKPOINT_PATH.value
  target_sample_rate = _TARGET_SAMPLE_RATE.value
  impaired_ohc_health = _IMPAIRED_OHC_HEALTH.value

  checkpointing.require_checkpoint(checkpoint_path)

  run_config = checkpointing.read_run_config(checkpoint_path)
  if run_config is None:
    logging.warning(
      'Checkpoint at %s has no config.json sidecar; defaulting to VikingModel.',
      checkpoint_path,
    )
    model: audio_model.AudioModel = viking_jax.VikingModel()
  else:
    if not flags.FLAGS['target_sample_rate'].present:
      target_sample_rate = run_config.target_sample_rate
    elif _TARGET_SAMPLE_RATE.value != run_config.target_sample_rate:
      logging.warning(
        'Flag --target_sample_rate=%d overrides checkpoint rate %d.',
        _TARGET_SAMPLE_RATE.value,
        run_config.target_sample_rate,
      )

    if not flags.FLAGS['impaired_ohc_health'].present:
      impaired_ohc_health = run_config.impaired_ohc_health
    elif not math.isclose(
      _IMPAIRED_OHC_HEALTH.value,
      run_config.impaired_ohc_health,
      rel_tol=1e-5,
      abs_tol=1e-5,
    ):
      logging.warning(
        'Flag --impaired_ohc_health=%f overrides checkpoint health %f.',
        _IMPAIRED_OHC_HEALTH.value,
        run_config.impaired_ohc_health,
      )

    if run_config.model:
      model = audio_model.from_config(run_config.model)
    else:
      model = viking_jax.VikingModel()

  skipped = report.evaluate_checkpoint(
    checkpoint_path=checkpoint_path,
    dataset_path=_INPUT_BASE_DIR.value,
    report_directory=_REPORT_DIRECTORY.value,
    num_analysis_clips=flag_utils.positive_or_none(_NUM_ANALYSIS_CLIPS.value),
    run_name=_RUN_NAME.value,
    target_sample_rate=target_sample_rate,
    impaired_ohc_health=impaired_ohc_health,
    num_nap_heatmaps=_NUM_NAP_HEATMAPS.value,
    model=model,
    device=jax.devices('cpu')[0] if _RESTORE_ON_CPU.value else None,
    seed=_NOISE_SEED.value,
    level_reference=typing.cast(
      transforms.LevelReference, _LEVEL_REFERENCE.value
    ),
  )

  if skipped:
    return 1
  return None


if __name__ == '__main__':
  app.run(main)
