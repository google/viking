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

"""End-to-end test that the Viking-CARFAC DAL training loop and CLI helpers run."""

import os
import tempfile

import jax
import numpy as np
from absl.testing import absltest
from etils import epath
from hp_acoustic.io import audio_io
from hp_acoustic.losses import waveform_losses
from hp_acoustic.training import callbacks as callbacks_lib
from hp_acoustic.training import checkpointing
from hp_acoustic.training import testing

from viking_jax import viking_jax
from viking_jax.scripts import flag_utils
from viking_jax.scripts import run_inference


class VikingTrainingLoopSmokeTest(absltest.TestCase):
  """End-to-end DAL training loop, checkpointing, and streaming inference."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._output_tmp = tempfile.TemporaryDirectory()
    cls._data_tmp = tempfile.TemporaryDirectory()
    cls.output_dir = cls._output_tmp.name
    cls.data_dir = os.path.join(cls._data_tmp.name, 'synthetic')
    testing.write_synthetic_dataset(cls.data_dir, split='train', seed=0)
    testing.write_synthetic_dataset(cls.data_dir, split='test', seed=1)

    cls.model = viking_jax.VikingModel(
      module=viking_jax.VikingGenerator(
        use_low_latency_filterbanks=True,
        convert_to_min_phase=True,
        shift_factor=32,
        num_subbands=64,
        prototype_filter_length=64,
        num_taps=32,
        lstm_depth=2,
        lstm_units=8,
        hub_units=4,
        num_parallel=2,
        max_mask_gain=20.0,
      ),
      init_shape=(1, testing.TRAINING_EXAMPLE_SAMPLES, 1),
    )

    run_config = checkpointing.RunConfig(
      target_sample_rate=testing.SAMPLE_RATE,
      impaired_ohc_health=0.5,
      model=cls.model.get_config(),
    )
    cls.smoke_results = testing.train_toy_run(
      output_dir=cls.output_dir,
      data_dir=cls.data_dir,
      model=cls.model,
      validate=True,
      run_name='viking_smoke',
      callbacks=[
        callbacks_lib.Checkpoint(
          output_dir=cls.output_dir,
          every=1,
          run_name='viking_smoke_cfg',
          run_config=run_config,
        )
      ],
    )

  @classmethod
  def tearDownClass(cls):
    cls._output_tmp.cleanup()
    cls._data_tmp.cleanup()
    super().tearDownClass()

  def test_one_epoch_completes_with_finite_carfac_losses(self):
    results = self.smoke_results
    self.assertIsNotNone(results)
    self.assertNotEmpty(results.history['g_total_loss'])
    self.assertNotEmpty(results.history['val_ssim_loss'])
    for name, series in results.history.items():
      for value in series:
        self.assertTrue(
          np.isfinite(value), f'loss {name} was {value}, not finite'
        )
    for leaf in jax.tree.leaves(results.params):
      self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))

  def test_checkpoint_restores_and_streaming_inference_matches_batch(self):
    ckpt = os.path.join(
      self.output_dir, 'viking_smoke_cfg', 'models', 'checkpoint_step_2'
    )
    self.assertTrue(epath.Path(ckpt).exists())

    restored_params = checkpointing.restore_params(ckpt, self.model)
    resumed = testing.train_toy_run(
      output_dir=self.output_dir,
      data_dir=self.data_dir,
      model=self.model,
      run_name='viking_resumed',
      init_params=restored_params,
      loss=(waveform_losses.L1WaveformLoss(weight=1.0),),
      max_steps=1,
    )
    self.assertIsNotNone(resumed)

    # Verify streaming WAV helper matches full-sequence `model.apply`
    wav_in = 0.05 * np.sin(
      2.0 * np.pi * 440.0 * np.arange(300, dtype=np.float32) / 16000.0
    )
    stream_out = run_inference._run_viking_streaming_wav(
      self.model.module, restored_params, wav_in
    )
    batch_out = np.asarray(
      self.model.apply(restored_params, wav_in[None, :, None])[0, :, 0]
    )
    self.assertEqual(stream_out.shape, wav_in.shape)
    np.testing.assert_allclose(stream_out, batch_out, atol=1e-4)

    out_wav_path = os.path.join(self.output_dir, 'enhanced.wav')
    audio_io.save_audio_to_file(stream_out, out_wav_path, testing.SAMPLE_RATE)
    reloaded, sr = audio_io.read_wav(out_wav_path, testing.SAMPLE_RATE)
    self.assertEqual(sr, testing.SAMPLE_RATE)
    self.assertEqual(len(reloaded), len(wav_in))

  def test_flag_utils_positive_or_none(self):
    self.assertIsNone(flag_utils.positive_or_none(None))
    self.assertIsNone(flag_utils.positive_or_none(0))
    self.assertIsNone(flag_utils.positive_or_none(-1))
    self.assertEqual(flag_utils.positive_or_none(25), 25)


if __name__ == '__main__':
  absltest.main()
