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

"""Unit tests for Viking JAX model."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from viking_jax import viking_jax

jax.config.update("jax_disable_jit", False)


def _run_viking_streaming(
    use_low_latency: bool = False,
    convert_to_min_phase: bool = False,
    lstm_depth: int = 1,
    num_taps: int = 32,
) -> float:
  if use_low_latency:
    shift = 16
    model = viking_jax.VikingGenerator(
        use_low_latency_filterbanks=True,
        shift_factor=shift,
        num_subbands=64,
        prototype_filter_length=64,
        num_taps=num_taps,
        lstm_units=8,
        hub_units=4,
        num_parallel=2,
        lstm_depth=lstm_depth,
        convert_to_min_phase=convert_to_min_phase,
    )
  else:
    shift = 16
    model = viking_jax.VikingGenerator(
        frame_length=64,
        frame_step=shift,
        lstm_units=8,
        hub_units=4,
        num_parallel=2,
        lstm_depth=lstm_depth,
    )

  batch = 1
  num_chunks = 16
  total_samples = num_chunks * shift

  key = jax.random.PRNGKey(42)
  x = jax.random.normal(key, (batch, total_samples, 1))
  rng = jax.random.PRNGKey(1)

  variables = model.init(rng, x, train=False)

  # 1. Run Full-Sequence Batch Mode directly on unpadded x
  out_batch = model.apply(variables, x, train=False)
  out_batch_arr = jnp.asarray(out_batch)

  # 2. Run In-Stream Chunk-by-Chunk Streaming Mode
  init_state = model.init_stream_state(batch_size=batch)

  @jax.jit
  def step_fn(state, chunk):
    return model.apply(
        variables,
        state,
        chunk,
        method=model.stream_step,
    )

  stream_state = init_state
  stream_chunks = []

  for i in range(num_chunks):
    chunk = x[:, i * shift : (i + 1) * shift, :]
    stream_state, out_chunk = step_fn(stream_state, chunk)
    stream_chunks.append(out_chunk)

  out_stream = jnp.concatenate(stream_chunks, axis=1)
  out_stream_arr = jnp.asarray(out_stream)

  max_diff = float(jnp.max(jnp.abs(out_batch_arr - out_stream_arr)))
  return max_diff


class VikingJaxTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_debug_nans", True)

  def test_viking_streaming_standard(self):
    max_diff = _run_viking_streaming(use_low_latency=False)
    self.assertLess(max_diff, 1e-4)

  def test_viking_streaming_low_latency_linear_phase(self):
    max_diff = _run_viking_streaming(
        use_low_latency=True, convert_to_min_phase=False
    )
    self.assertLess(max_diff, 1e-4)

  def test_viking_streaming_low_latency_min_phase(self):
    max_diff = _run_viking_streaming(
        use_low_latency=True, convert_to_min_phase=True
    )
    self.assertLess(max_diff, 1e-4)

  def test_viking_streaming_multi_layer_lstm(self):
    max_diff = _run_viking_streaming(
        use_low_latency=True, convert_to_min_phase=True, lstm_depth=2
    )
    self.assertLess(max_diff, 1e-4)

  def test_viking_streaming_custom_num_taps(self):
    for convert_to_min_phase in (False, True):
      max_diff = _run_viking_streaming(
          use_low_latency=True,
          convert_to_min_phase=convert_to_min_phase,
          num_taps=16,
      )
      self.assertLess(max_diff, 1e-4)

  def test_stream_step_invalid_chunk_size_raises(self):
    model = viking_jax.VikingGenerator(frame_length=64, frame_step=16)
    x = jnp.zeros((1, 64, 1))
    variables = model.init(jax.random.PRNGKey(0), x, train=False)
    state = model.init_stream_state(batch_size=1)
    bad_chunk = jnp.zeros((1, 12, 1))
    with self.assertRaises(ValueError):
      model.apply(variables, state, bad_chunk, method=model.stream_step)

  def test_window_functions(self):
    for window_fn in (viking_jax.hann_window, viking_jax.hamming_window):
      win_sym = window_fn(64, periodic=False)
      win_per = window_fn(64, periodic=True)
      self.assertEqual(win_sym.shape, (64,))
      self.assertEqual(win_per.shape, (64,))
      self.assertTrue(jnp.allclose(win_sym, win_sym[::-1], atol=1e-6))
      self.assertTrue(jnp.allclose(win_per[1:], win_per[1:][::-1], atol=1e-6))

  def test_overlap_and_add_and_inverse_stft_window(self):
    batch = 2
    frame_length = 64
    frame_step = 16
    frames = 12
    expected_len = frame_length + frame_step * (frames - 1)

    dummy_frames = jnp.ones((batch, frames, frame_length))
    ola_out = viking_jax.overlap_and_add(dummy_frames, frame_step)
    self.assertEqual(ola_out.shape, (batch, expected_len))

    key = jax.random.PRNGKey(0)
    signal = jax.random.normal(key, (batch, expected_len))
    framed = jax.lax.conv_general_dilated_patches(
        signal[..., None],
        (frame_length,),
        (frame_step,),
        padding="VALID",
        dimension_numbers=("NTC", "OIT", "NTC"),
    )
    fwd_win = viking_jax.hann_window(frame_length)
    inv_win = viking_jax.inverse_stft_window_fn(
        frame_length, frame_step, viking_jax.hann_window
    )
    stft = jnp.fft.rfft(framed * fwd_win, frame_length)
    istft_frames = jnp.fft.irfft(stft, frame_length) * inv_win
    reconstructed = viking_jax.overlap_and_add(istft_frames, frame_step)

    interior = slice(frame_length, expected_len - frame_length)
    self.assertTrue(
        jnp.allclose(reconstructed[:, interior], signal[:, interior], atol=1e-5)
    )

  def test_prototype_filter_and_min_phase_conversion(self):
    proto = viking_jax.design_prototype_filter(512)
    self.assertEqual(proto.shape, (512,))
    self.assertTrue(jnp.allclose(proto, proto[::-1], atol=1e-4))

    num_taps = 256
    freqs = jnp.arange(num_taps // 2 + 1, dtype=jnp.float32)
    mask = 0.2 + 0.8 * jnp.exp(-((freqs / 32.0) ** 2))
    h_lin = jnp.fft.fftshift(
        jnp.fft.irfft(mask, num_taps)
    ) * viking_jax.hamming_window(num_taps, periodic=False)
    h_min = viking_jax.convert_fir_taps_to_min_phase_jax(h_lin, num_taps)
    self.assertEqual(h_min.shape, (num_taps,))

    mag_lin = jnp.abs(jnp.fft.rfft(h_lin, n=num_taps * 2))
    mag_min = jnp.abs(jnp.fft.rfft(h_min, n=num_taps * 2))
    self.assertTrue(jnp.allclose(mag_lin, mag_min, atol=1e-5))

    peak_lin = int(jnp.argmax(jnp.abs(h_lin)))
    peak_min = int(jnp.argmax(jnp.abs(h_min)))
    self.assertLess(peak_min, peak_lin)

  def test_resolve_num_taps_and_synthesis_fir_taps_helper(self):
    filter_len = 64
    model_small = viking_jax.VikingGenerator(
        prototype_filter_length=filter_len, num_taps=24
    )
    model_capped = viking_jax.VikingGenerator(
        prototype_filter_length=filter_len, num_taps=128
    )
    self.assertEqual(viking_jax._resolve_num_taps(model_small, filter_len), 24)
    self.assertEqual(viking_jax._resolve_num_taps(model_capped, filter_len), 32)

    proto = viking_jax.design_prototype_filter(filter_len)
    mask_3d = jax.random.uniform(
        jax.random.PRNGKey(5),
        (2, 3, filter_len // 2 + 1),
        minval=0.1,
        maxval=2.0,
    )
    for convert_to_min_phase in (False, True):
      hparams = viking_jax.VikingGenerator(
          use_low_latency_filterbanks=True,
          shift_factor=16,
          num_subbands=filter_len,
          prototype_filter_length=filter_len,
          num_taps=24,
          convert_to_min_phase=convert_to_min_phase,
      )
      taps_3d = viking_jax._compute_synthesis_fir_taps(mask_3d, proto, hparams)
      taps_2d = viking_jax._compute_synthesis_fir_taps(
          mask_3d[:, 1, :], proto, hparams
      )
      self.assertEqual(taps_3d.shape, (2, 3, 24))
      self.assertEqual(taps_2d.shape, (2, 24))
      self.assertTrue(jnp.allclose(taps_3d[:, 1, :], taps_2d, atol=1e-6))

  def test_low_latency_filterbanks(self):
    batch = 2
    samples = 256
    shift = 32
    filter_len = 128
    hparams = viking_jax.VikingGenerator(
        use_low_latency_filterbanks=True,
        shift_factor=shift,
        num_subbands=filter_len,
        prototype_filter_length=filter_len,
        num_taps=filter_len // 2,
    )
    prototype_filter = viking_jax.design_prototype_filter(filter_len)
    x_audio = jax.random.normal(jax.random.PRNGKey(3), (batch, samples))

    spec = viking_jax.low_latency_analysis_filterbank(
        x_audio, prototype_filter, hparams
    )
    expected_frames = (samples + shift - 1) // shift
    expected_bins = filter_len // 2 + 1
    self.assertEqual(spec.shape, (batch, expected_frames, expected_bins))
    self.assertTrue(jnp.all(jnp.isfinite(spec)))

    unity_mask = jnp.ones_like(spec)
    y_audio = viking_jax.low_latency_synthesis_filterbank(
        x_audio, prototype_filter, unity_mask, hparams
    )
    self.assertEqual(y_audio.shape, (batch, samples))
    self.assertTrue(jnp.all(jnp.isfinite(y_audio)))

  def test_viking_cell(self):
    batch = 2
    num_parallel = 4
    lstm_units = 32
    hub_units = 8
    num_bins = 129
    x_t = jax.random.normal(jax.random.PRNGKey(7), (batch, num_bins))

    for lstm_depth, max_mask_gain in ((1, 20.0), (2, 1.0)):
      cell = viking_jax.VikingCell(
          lstm_units=lstm_units,
          hub_units=hub_units,
          num_bins=num_bins,
          num_parallel=num_parallel,
          lstm_depth=lstm_depth,
          max_mask_gain=max_mask_gain,
      )
      carry = tuple(
          (jnp.zeros((batch, lstm_units)), jnp.zeros((batch, lstm_units)))
          for _ in range(lstm_depth * num_parallel)
      )
      variables = cell.init(jax.random.PRNGKey(8), carry, x_t)
      new_carry, mask = cell.apply(variables, carry, x_t)

      self.assertLen(new_carry, lstm_depth * num_parallel)
      for c, h in new_carry:
        self.assertEqual(c.shape, (batch, lstm_units))
        self.assertEqual(h.shape, (batch, lstm_units))
      self.assertEqual(mask.shape, (batch, num_bins))
      self.assertTrue(jnp.all((mask > 0.0) & (mask < max_mask_gain)))

  def test_viking_cell_unity_gain_initialization(self):
    batch = 1
    num_parallel = 2
    lstm_units = 8
    hub_units = 4
    num_bins = 33
    x_zero = jnp.zeros((batch, num_bins))
    carry = tuple(
        (jnp.zeros((batch, lstm_units)), jnp.zeros((batch, lstm_units)))
        for _ in range(num_parallel)
    )

    cell_20 = viking_jax.VikingCell(
        lstm_units=lstm_units,
        hub_units=hub_units,
        num_bins=num_bins,
        num_parallel=num_parallel,
        max_mask_gain=20.0,
    )
    vars_20 = cell_20.init(jax.random.PRNGKey(9), carry, x_zero)
    _, mask_20 = cell_20.apply(vars_20, carry, x_zero)
    self.assertTrue(jnp.allclose(mask_20, 1.0, atol=1e-5))

    cell_1 = viking_jax.VikingCell(
        lstm_units=lstm_units,
        hub_units=hub_units,
        num_bins=num_bins,
        num_parallel=num_parallel,
        max_mask_gain=1.0,
    )
    vars_1 = cell_1.init(jax.random.PRNGKey(9), carry, x_zero)
    _, mask_1 = cell_1.apply(vars_1, carry, x_zero)
    self.assertTrue(jnp.allclose(mask_1, 0.5, atol=1e-5))

  def test_viking_generator_shapes_and_gradients(self):
    batch = 2
    samples = 128
    key = jax.random.PRNGKey(11)
    x_3d = jax.random.normal(key, (batch, samples, 1))
    x_2d = jnp.squeeze(x_3d, axis=-1)

    for use_low_latency in (False, True):
      model = viking_jax.VikingGenerator(
          frame_length=64,
          frame_step=16,
          use_low_latency_filterbanks=use_low_latency,
          shift_factor=32,
          num_subbands=64,
          prototype_filter_length=64,
          num_taps=32,
          lstm_units=8,
          hub_units=4,
          num_parallel=2,
      )
      variables = model.init(jax.random.PRNGKey(12), x_3d, train=True)
      out_3d = model.apply(variables, x_3d, train=False)
      out_2d = model.apply(variables, x_2d, train=False)
      self.assertEqual(out_3d.shape, (batch, samples, 1))
      self.assertEqual(out_2d.shape, (batch, samples, 1))
      self.assertTrue(jnp.allclose(out_3d, out_2d, atol=1e-6))

      def loss_fn(params, m=model):
        pred = m.apply({"params": params}, x_3d, train=True)
        return jnp.mean(jnp.abs(pred - x_3d))

      grads = jax.grad(loss_fn)(variables["params"])
      grad_leaves = jax.tree_util.tree_leaves(grads)
      self.assertTrue(all(jnp.all(jnp.isfinite(g)) for g in grad_leaves))
      total_grad_norm = sum(float(jnp.sum(jnp.abs(g))) for g in grad_leaves)
      self.assertGreater(total_grad_norm, 0.0)

  def test_public_api_exports(self):
    import viking_jax as pkg  # pylint: disable=g-import-not-at-top

    expected_exports = {
        "VikingStreamState",
        "VikingCell",
        "VikingGenerator",
        "hann_window",
        "hamming_window",
        "design_prototype_filter",
        "convert_fir_taps_to_min_phase_jax",
        "low_latency_analysis_filterbank",
        "low_latency_synthesis_filterbank",
    }
    self.assertEqual(set(pkg.__all__), expected_exports)
    for name in expected_exports:
      self.assertTrue(hasattr(pkg, name))


if __name__ == "__main__":
  absltest.main()
