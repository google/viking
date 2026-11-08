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

"""Head-to-head DAL CARFAC benchmark: Unprocessed vs. SHACA vs. SEANet vs. Viking.

Evaluates SHACA (prescribed NAL-NL2/CAM2 WDRC and DAL-trained), SEANet, and
Viking (minimum-phase and linear-phase) under identical CARFAC SAI/NAP
objectives (arXiv:2606.04103), reporting parameter counts, algorithmic latency
(ms), and auditory-domain losses.

If `--input_base_dir` is omitted, a temporary synthetic corpus is generated so
you can smoke-test the full comparison immediately.
"""

import os
import tempfile
from collections.abc import Mapping
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import optax
import shaca_jax
from absl import app
from absl import flags
from hp_acoustic.data import clips as data_clips
from hp_acoustic.data import pipeline
from hp_acoustic.data import records
from hp_acoustic.data import transforms
from hp_acoustic.data.sources import color_noise
from hp_acoustic.data.sources import processed
from hp_acoustic.losses import representation_losses
from hp_acoustic.losses import terms as terms_lib
from hp_acoustic.losses import waveform_losses
from hp_acoustic.models import audio_model
from hp_acoustic.models import carfac
from hp_acoustic.models import cochlea
from hp_acoustic.models import context as auditory_context
from hp_acoustic.training import data as data_lib
from hp_acoustic.training import loop
from hp_acoustic.training import steps
from seanet.models import seanet

from viking_jax import viking_jax

_INPUT_BASE_DIR = flags.DEFINE_string(
  'input_base_dir',
  None,
  'Prepared corpus directory (train/ and test/). If None, uses synthetic audio.',
)
_TARGET_SAMPLE_RATE = flags.DEFINE_integer(
  'target_sample_rate', 16000, 'Sample rate in Hz (e.g. 16000 or 32000).'
)
_CHUNK_SECONDS = flags.DEFINE_float(
  'chunk_seconds',
  0.128,
  'Audio chunk duration in seconds (multiple of 256 samples).',
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 4, 'Global batch size.')
_MAX_STEPS = flags.DEFINE_integer(
  'max_steps', 10, 'Number of Adam optimization steps per model.'
)
_LEARNING_RATE = flags.DEFINE_float(
  'learning_rate', 2e-3, 'Adam learning rate for Viking.'
)
_SEANET_LEARNING_RATE = flags.DEFINE_float(
  'seanet_learning_rate', 2e-4, 'Adam learning rate for SEANet.'
)
_INIT_MASK_GAIN = flags.DEFINE_float(
  'init_mask_gain',
  2.5,
  'Initial insertion gain for Viking on impaired CARFAC (OHC=0.5).',
)
_SHACA_LEARNING_RATE = flags.DEFINE_float(
  'shaca_learning_rate',
  5e-2,
  'Adam learning rate for SHACA unconstrained WDRC logits.',
)
_IMPAIRED_OHC_HEALTH = flags.DEFINE_float(
  'impaired_ohc_health', 0.5, 'Impaired CARFAC OHC health (0.0 to 1.0).'
)
_OUTPUT_PNG = flags.DEFINE_string(
  'output_png',
  None,
  'Optional path to save a summary PNG chart of the benchmark results.',
)


class _IdentityAudioModel:
  """Unprocessed noisy pass-through baseline."""

  @property
  def name(self) -> str:
    return 'unprocessed'

  def init(self, rng: jax.Array) -> dict[str, jax.Array]:
    del rng
    return {}

  def apply(
    self, params: Mapping[str, jax.Array], audio: jnp.ndarray
  ) -> jnp.ndarray:
    del params
    return audio

  def constrain(
    self, params: Mapping[str, jax.Array]
  ) -> Mapping[str, jax.Array]:
    return params

  def get_config(self) -> Mapping[str, object]:
    return {'name': self.name}


def _count_params(params) -> int:
  return sum(int(x.size) for x in jax.tree.leaves(params))


def _write_harmonic_speechlike_corpus(
  data_dir: str, sample_rate: int, num_samples: int
) -> None:
  """Creates synthetic AM-modulated harmonic clips for fast local benchmarking."""
  t = jnp.arange(num_samples, dtype=jnp.float32) / sample_rate
  for split, base_seed in (('train', 10), ('test', 20)):
    clips = []
    for i in range(8):
      f0 = 120.0 + 25.0 * i
      harmonics = sum(
        (1.0 / k) * jnp.sin(2.0 * jnp.pi * (k * f0) * t) for k in range(1, 7)
      )
      env = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * 3.0 * t))
      jitter = 0.01 * jax.random.normal(
        jax.random.PRNGKey(base_seed + i), (num_samples,)
      )
      waveform = 0.05 * env * harmonics + jitter
      clips.append(
        data_clips.Clip(
          audio=waveform,
          sample_rate=sample_rate,
          clip_id=f'{split}_{i}.wav',
        )
      )
    records.write_clips(clips, os.path.join(data_dir, split))


def main(argv: Sequence[str]) -> None:
  del argv
  sr = _TARGET_SAMPLE_RATE.value
  chunk_sec = _CHUNK_SECONDS.value
  hop_2ms = max(16, round(0.002 * sr))

  tmp_ctx = (
    tempfile.TemporaryDirectory() if _INPUT_BASE_DIR.value is None else None
  )
  try:
    if tmp_ctx is not None:
      corpus_dir = os.path.join(tmp_ctx.name, 'synthetic_corpus')
      chunk_samples = round(chunk_sec * sr)
      _write_harmonic_speechlike_corpus(corpus_dir, sr, chunk_samples * 2)
    else:
      corpus_dir = _INPUT_BASE_DIR.value

    data_cfg = pipeline.DataConfig(
      sources=(processed.ProcessedClips(corpus_dir),),
      transforms=(
        transforms.BlendSource(
          sources=(color_noise.ColorNoise(color='white', sample_rate=sr),),
          snr_db=(-2.0, 5.0),
          num_sources=(1, 1),
        ),
        transforms.NormalizeLevel(level_db_spl=(60.0, 75.0)),
        transforms.CosineWindow(ramp_seconds=0.005),
      ),
      global_batch_size=_BATCH_SIZE.value,
      sample_rate=sr,
      chunk_seconds=chunk_sec,
      max_epochs=50,
      seed=42,
    )

    cochleae = cochlea.CochleaPair(
      healthy=carfac.build(sr),
      impaired=carfac.build(sr, ohc_health=_IMPAIRED_OHC_HEALTH.value),
    )
    ctx = auditory_context.build(cochleae)

    train_terms: tuple[terms_lib.LossTerm, ...] = (
      representation_losses.SaiSsimLoss(weight=1.0, timewinsz=36, chanwinsz=6),
      representation_losses.SaiPndLoss(weight=1.0, alpha=0.5, beta=0.8),
      representation_losses.NapL1Loss(weight=1.0),
      waveform_losses.LowFrequencyDriftLoss(weight=2.0, sample_rate=sr),
    )
    eval_terms: tuple[terms_lib.LossTerm, ...] = train_terms

    val_batches = list(
      data_lib.validation_batches(data_cfg, num_val_examples=8)
    )

    # Evaluate unprocessed noisy baseline
    baseline = _IdentityAudioModel()
    base_eval_fn = steps.build_eval_step(baseline.apply, eval_terms, ctx)
    base_metrics = loop.run_validation_epoch(
      {}, val_batches, base_eval_fn, eval_terms
    )

    # Evaluate SHACA out-of-the-box clinical prescription (0 steps)
    shaca_frame_len = 16 if sr <= 16000 else 32
    shaca_model = shaca_jax.SHACAModel(
      sample_rate_hz=float(sr),
      frame_len=shaca_frame_len,
      init_mode='prescription',
      freeze_thresholds=False,
    )
    shaca_init_params = shaca_model.init(jax.random.PRNGKey(0))
    shaca_eval_fn = steps.build_eval_step(shaca_model.apply, eval_terms, ctx)
    shaca_presc_metrics = loop.run_validation_epoch(
      shaca_init_params, val_batches, shaca_eval_fn, eval_terms
    )

    candidates: list[tuple[str, float, float, audio_model.AudioModel]] = [
      (
        'SHACA (DAL-Trained)',
        shaca_model.latency_ms,
        _SHACA_LEARNING_RATE.value,
        shaca_model,
      ),
      (
        'SEANet (causal conv)',
        1000.0 * seanet.total_stride(seanet.SEANetGenerator()) / sr,
        _SEANET_LEARNING_RATE.value,
        seanet.SEANetModel(),
      ),
      (
        'Viking (min-phase)',
        1000.0 * hop_2ms / sr,
        _LEARNING_RATE.value,
        viking_jax.VikingModel(
          module=viking_jax.VikingGenerator(
            use_low_latency_filterbanks=True,
            convert_to_min_phase=True,
            shift_factor=hop_2ms,
            num_subbands=256 if sr <= 16000 else 512,
            prototype_filter_length=256 if sr <= 16000 else 512,
            num_taps=128 if sr <= 16000 else 256,
            lstm_depth=2,
            lstm_units=32,
            hub_units=8,
            max_mask_gain=20.0,
            init_mask_gain=_INIT_MASK_GAIN.value,
          )
        ),
      ),
      (
        'Viking (linear-phase)',
        1000.0 * (hop_2ms + (128 if sr <= 16000 else 256) // 2) / sr,
        _LEARNING_RATE.value,
        viking_jax.VikingModel(
          module=viking_jax.VikingGenerator(
            use_low_latency_filterbanks=True,
            convert_to_min_phase=False,
            shift_factor=hop_2ms,
            num_subbands=256 if sr <= 16000 else 512,
            prototype_filter_length=256 if sr <= 16000 else 512,
            num_taps=128 if sr <= 16000 else 256,
            lstm_depth=2,
            lstm_units=32,
            hub_units=8,
            max_mask_gain=20.0,
            init_mask_gain=_INIT_MASK_GAIN.value,
          )
        ),
      ),
    ]

    rows = [
      (
        'Unprocessed (Impaired)',
        0,
        0.0,
        base_metrics['val_ssim_loss'],
        base_metrics['val_pnd_sai_loss'],
        base_metrics['val_l1_nap_loss'],
      ),
      (
        'SHACA (Prescribed)',
        _count_params(shaca_init_params),
        shaca_model.latency_ms,
        shaca_presc_metrics['val_ssim_loss'],
        shaca_presc_metrics['val_pnd_sai_loss'],
        shaca_presc_metrics['val_l1_nap_loss'],
      ),
    ]

    for label, latency_ms, lr, model in candidates:
      optimizer = optax.adam(lr, b1=0.5, b2=0.9)
      res = loop.fit(
        model=model,
        data=pipeline.build(data_cfg),
        loss=train_terms,
        context=ctx,
        optimizer=optimizer,
        max_steps=_MAX_STEPS.value,
        eval_every=_MAX_STEPS.value,
        val_data=val_batches,
        seed=0,
      )
      eval_fn = steps.build_eval_step(model.apply, eval_terms, ctx)
      m = loop.run_validation_epoch(
        res.params, val_batches, eval_fn, eval_terms
      )
      rows.append(
        (
          label,
          _count_params(res.params),
          latency_ms,
          m['val_ssim_loss'],
          m['val_pnd_sai_loss'],
          m['val_l1_nap_loss'],
        )
      )

    print(
      f'\n=== DAL CARFAC Benchmark ({sr} Hz, OHC={_IMPAIRED_OHC_HEALTH.value}, '
      f'steps={_MAX_STEPS.value}) ==='
    )
    print(
      f"{'Model':<24} | {'Params':>9} | {'Latency':>8} | "
      f"{'SAI SSIM':>9} | {'SAI PND':>9} | {'NAP L1':>9}"
    )
    print('-' * 82)
    for label, n_params, lat_ms, ssim_l, pnd_l, nap_l in rows:
      print(
        f'{label:<24} | {n_params:>9,d} | {lat_ms:>6.2f}ms | '
        f'{ssim_l:>9.4f} | {pnd_l:>9.4f} | {nap_l:>9.4f}'
      )

    if _OUTPUT_PNG.value:
      from viking_jax.scripts import generate_comparison_figures as fig_lib  # noqa: PLC0415  # pylint: disable=g-import-not-at-top

      os.makedirs(
        os.path.dirname(os.path.abspath(_OUTPUT_PNG.value)), exist_ok=True
      )
      dict_rows = [
        {
          'label': (
            'Viking (Min-Phase)'
            if label == 'Viking (min-phase)'
            else 'Viking (Linear-Phase)'
            if label == 'Viking (linear-phase)'
            else 'SEANet (Causal Conv)'
            if label == 'SEANet (causal conv)'
            else label
          ),
          'params': n_params,
          'latency_ms': lat_ms,
          'ssim': float(ssim_l),
          'pnd': float(pnd_l),
          'nap_l1': float(nap_l),
          'aud_total': float(ssim_l + pnd_l + nap_l),
        }
        for label, n_params, lat_ms, ssim_l, pnd_l, nap_l in rows
      ]
      fig_lib._plot_fig1_overview(  # pylint: disable=protected-access
        dict_rows,
        sr,
        _IMPAIRED_OHC_HEALTH.value,
        _MAX_STEPS.value,
        _OUTPUT_PNG.value,
      )
      print(f'\nSaved summary figure to {_OUTPUT_PNG.value}')
  finally:
    if tmp_ctx is not None:
      tmp_ctx.cleanup()


if __name__ == '__main__':
  app.run(main)
