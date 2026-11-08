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

"""Tests for `VikingModel` / `VikingAudioModel` and `hp_acoustic` integration."""

import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from hp_acoustic.losses import representation_losses
from hp_acoustic.losses import waveform_losses
from hp_acoustic.models import audio_model
from hp_acoustic.models import carfac
from hp_acoustic.models import cochlea
from hp_acoustic.models import conformance
from hp_acoustic.models import context as auditory_context
from hp_acoustic.training import checkpointing
from hp_acoustic.training import steps

from viking_jax import audio_model as viking_audio_model
from viking_jax import viking_jax


class VikingAudioModelTest(absltest.TestCase):
  """Tests verifying `VikingModel` AudioModel conformance and DAL integration."""

  def test_conforms_to_audio_model_stft_and_low_latency(self):
    sample = jnp.zeros((2, 512, 1), dtype=jnp.float32)

    stft_model = viking_jax.VikingModel(
      module=viking_jax.VikingGenerator(
        frame_length=64,
        frame_step=16,
        lstm_units=8,
        hub_units=4,
        num_parallel=2,
      )
    )
    self.assertIsInstance(stft_model, audio_model.AudioModel)
    conformance.check(stft_model, sample_input=sample)

    low_lat_model = viking_jax.VikingModel(
      module=viking_jax.VikingGenerator(
        use_low_latency_filterbanks=True,
        shift_factor=16,
        num_subbands=64,
        prototype_filter_length=64,
        num_taps=32,
        convert_to_min_phase=True,
        lstm_depth=2,
        lstm_units=8,
        hub_units=4,
        num_parallel=2,
      )
    )
    self.assertIsInstance(low_lat_model, audio_model.AudioModel)
    conformance.check(low_lat_model, sample_input=sample)

  def test_audio_model_module_exports(self):
    self.assertIs(viking_audio_model.VikingModel, viking_jax.VikingModel)
    self.assertIs(viking_audio_model.VikingAudioModel, viking_jax.VikingModel)
    self.assertIs(
      viking_audio_model.VikingGenerator, viking_jax.VikingGenerator
    )

  def test_matches_calling_generator_directly(self):
    module = viking_jax.VikingGenerator(
      use_low_latency_filterbanks=True,
      shift_factor=16,
      num_subbands=64,
      prototype_filter_length=64,
      num_taps=32,
      convert_to_min_phase=True,
      lstm_units=8,
      hub_units=4,
      num_parallel=2,
    )
    audio = jax.random.normal(jax.random.PRNGKey(3), (2, 256, 1))
    key = jax.random.PRNGKey(0)

    direct_params = module.init(key, jnp.zeros((1, 1024, 1)), train=False)[
      'params'
    ]
    direct = module.apply({'params': direct_params}, audio, train=False)

    adapter = viking_jax.VikingModel(module=module, init_shape=(1, 1024, 1))
    adapted = adapter.apply(adapter.init(key), audio)

    np.testing.assert_allclose(
      np.asarray(direct), np.asarray(adapted), atol=1e-6
    )

  def test_constrain_is_identity(self):
    adapter = viking_jax.VikingModel(
      module=viking_jax.VikingGenerator(
        frame_length=64,
        frame_step=16,
        lstm_units=8,
        hub_units=4,
        num_parallel=2,
      )
    )
    params = adapter.init(jax.random.PRNGKey(0))
    self.assertIs(adapter.constrain(params), params)

  def test_get_config_and_from_config_roundtrip(self):
    model = viking_jax.VikingModel(
      module=viking_jax.VikingGenerator(
        use_low_latency_filterbanks=True,
        shift_factor=32,
        num_subbands=256,
        prototype_filter_length=256,
        convert_to_min_phase=True,
        num_taps=128,
        num_parallel=4,
        lstm_depth=2,
        lstm_units=16,
        hub_units=8,
        in_magnitude_compression=0.25,
        max_mask_gain=15.0,
      ),
      init_shape=(1, 512, 1),
    )
    config = model.get_config()
    self.assertEqual(config['name'], 'viking')
    self.assertEqual(config['init_shape'], (1, 512, 1))
    self.assertEqual(config['lstm_depth'], 2)
    self.assertEqual(config['max_mask_gain'], 15.0)

    # Simulate JSON sidecar serialization (`init_shape` becomes a list).
    json_config = json.loads(json.dumps(config))
    restored = audio_model.from_config(json_config)
    self.assertIsInstance(restored, viking_jax.VikingModel)
    self.assertEqual(restored.name, 'viking')
    self.assertEqual(restored.init_shape, (1, 512, 1))
    self.assertEqual(restored.get_config(), config)

    with self.assertRaisesRegex(ValueError, 'init_shape'):
      viking_jax.VikingModel.from_config({'name': 'viking', 'init_shape': [1]})

  def test_checkpoint_save_and_restore_roundtrip(self):
    model = viking_jax.VikingModel(
      module=viking_jax.VikingGenerator(
        use_low_latency_filterbanks=True,
        shift_factor=16,
        num_subbands=64,
        prototype_filter_length=64,
        num_taps=32,
        lstm_units=8,
        hub_units=4,
        num_parallel=2,
      )
    )
    optimizer = optax.adam(1e-3)
    params = model.init(jax.random.PRNGKey(0))
    opt_state = optimizer.init(params)

    with tempfile.TemporaryDirectory() as tmpdir:
      run_cfg = checkpointing.RunConfig(
        target_sample_rate=16000,
        impaired_ohc_health=0.5,
        model=model.get_config(),
      )
      checkpointing.save_model(
        params=params,
        opt_state=opt_state,
        output_base_dir=tmpdir,
        run_name='viking_run',
        epoch_name='step_1',
        run_config=run_cfg,
      )
      ckpt_dir = os.path.join(
        tmpdir, 'viking_run', 'models', 'checkpoint_step_1'
      )
      restored_params = checkpointing.restore_params(ckpt_dir, model)
      jax.tree.map(
        np.testing.assert_allclose,
        restored_params,
        params,
      )
      restored_params_2, _ = checkpointing.restore_model(
        ckpt_dir, model, optimizer
      )
      jax.tree.map(
        np.testing.assert_allclose,
        restored_params_2,
        params,
      )

  def test_carfac_dal_train_step_with_viking_model(self):
    sample_rate = 16000
    model = viking_jax.VikingModel(
      module=viking_jax.VikingGenerator(
        use_low_latency_filterbanks=True,
        shift_factor=16,
        num_subbands=64,
        prototype_filter_length=64,
        num_taps=32,
        convert_to_min_phase=True,
        lstm_units=8,
        hub_units=4,
        num_parallel=2,
      ),
      init_shape=(1, 256, 1),
    )
    cochleae = cochlea.CochleaPair(
      healthy=carfac.build(sample_rate),
      impaired=carfac.build(sample_rate, ohc_health=0.5),
    )
    terms = (
      representation_losses.NapL1Loss(weight=1.0),
      waveform_losses.L1WaveformLoss(weight=0.1),
    )
    optimizer = optax.adam(1e-3)
    params = model.init(jax.random.PRNGKey(0))
    opt_state = optimizer.init(params)

    train_step = steps.build_train_step(
      generator_apply_fn=model.apply,
      g_optimizer_update_fn=optimizer.update,
      constrain_fn=model.constrain,
      terms=terms,
      context=auditory_context.build(cochleae),
    )

    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    clean = 0.05 * jax.random.normal(k1, (2, 256, 1))
    noisy = clean + 0.02 * jax.random.normal(k2, (2, 256, 1))
    batch = steps.Batch(noisy=noisy, clean=clean)

    new_params, _, losses_dict = train_step(params, opt_state, batch)
    for loss_name, loss_val in losses_dict.items():
      self.assertTrue(
        np.isfinite(float(loss_val)),
        f'Loss {loss_name} was non-finite: {loss_val}',
      )
    diffs = [
      float(jnp.max(jnp.abs(a - b)))
      for a, b in zip(
        jax.tree.leaves(new_params), jax.tree.leaves(params), strict=True
      )
    ]
    self.assertGreater(max(diffs), 0.0)


if __name__ == '__main__':
  absltest.main()
