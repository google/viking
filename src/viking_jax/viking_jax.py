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

"""Viking JAX Class Definition.

Simplified JAX/Flax implementation of the Viking low-latency speech enhancement
and hearing-aid model.

Reference:
  "Towards Sub-millisecond Latency Real-Time Speech Enhancement Models on
  Hearables" (arXiv:2409.18239).
"""

import math
from typing import Callable

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class VikingStreamState:
  """Container for in-stream state across audio chunks."""

  audio_buffer: jnp.ndarray
  lstm_carry: tuple[tuple[jnp.ndarray, jnp.ndarray], ...]
  overlap_buffer: jnp.ndarray | None = None
  filter_old: jnp.ndarray | None = None


def hann_window(frame_length: int, periodic: bool = True) -> jnp.ndarray:
  """Hanning window function."""
  if periodic and frame_length % 2 == 0:
    return jnp.hanning(frame_length + 1)[:-1]
  return jnp.hanning(frame_length)


def hamming_window(frame_length: int, periodic: bool = True) -> jnp.ndarray:
  """Hamming window function."""
  if periodic and frame_length % 2 == 0:
    return jnp.hamming(frame_length + 1)[:-1]
  return jnp.hamming(frame_length)


def overlap_and_add(signal: jnp.ndarray, frame_step: int) -> jnp.ndarray:
  """Overlap and add implementation in JAX."""
  batch, frames, frame_length = signal.shape
  overlaps = -(-frame_length // frame_step)
  signal = jnp.pad(
      signal, [(0, 0), (0, overlaps), (0, overlaps * frame_step - frame_length)]
  )
  signal = jnp.reshape(signal, [batch, frames + overlaps, overlaps, frame_step])
  signal = jnp.transpose(signal, [0, 2, 1, 3])
  signal = jnp.reshape(
      signal, [batch, overlaps * (frames + overlaps), frame_step]
  )
  extended = frames + overlaps - 1
  signal = signal[:, : overlaps * extended, :]
  signal = jnp.reshape(signal, [batch, overlaps, extended, frame_step])
  signal = jnp.sum(signal, axis=1)
  signal = jnp.reshape(signal, [batch, extended * frame_step])
  output_length = frame_length + frame_step * (frames - 1)
  return signal[:, :output_length]


def inverse_stft_window_fn(
    frame_length: int,
    frame_step: int,
    window_fn: Callable[[int], jnp.ndarray],
) -> jnp.ndarray:
  """Normalizes window for inverse STFT."""
  fwd_window = window_fn(frame_length).astype(jnp.float32)
  denom = fwd_window**2
  overlaps = -(-frame_length // frame_step)
  denom = jnp.pad(denom, [(0, overlaps * frame_step - frame_length)])
  denom = jnp.reshape(denom, [overlaps, frame_step])
  denom = jnp.sum(denom, axis=0, keepdims=True)
  denom = jnp.tile(denom, [overlaps, 1])
  denom = jnp.reshape(denom, [overlaps * frame_step])
  return fwd_window / (denom[:frame_length] + 1e-8)


def design_prototype_filter(filter_length: int = 512) -> jnp.ndarray:
  """Designs the prototype filter (Hamming-windowed sinc)."""
  index = jnp.linspace(0, filter_length - 1, filter_length, endpoint=True)
  group_delay = filter_length // 2
  # Avoid division by zero in sinc if index == group_delay
  prototype_filter = (1 / filter_length) * jnp.sinc(
      (index - group_delay) / filter_length
  )
  return hamming_window(filter_length, periodic=False) * prototype_filter


def convert_fir_taps_to_min_phase_jax(
    fir_taps: jnp.ndarray, num_taps: int
) -> jnp.ndarray:
  """Converts linear-phase FIR filter taps to minimum-phase via real cepstrum.

  Applies the causal Hilbert-transform window in the cepstral domain
  (arXiv:2409.18239) so that the resulting filter preserves the magnitude
  response of `fir_taps` while concentrating energy toward tap 0.

  Args:
    fir_taps: Input FIR filter taps of shape `[..., num_taps]`.
    num_taps: Number of FIR taps to output.

  Returns:
    Minimum-phase FIR filter taps of shape `[..., num_taps]`.
  """
  pad_width = [(0, 0)] * (fir_taps.ndim - 1) + [(0, num_taps)]
  fir_coefficients_padded = jnp.pad(fir_taps, pad_width, 'constant')

  h_k = jnp.fft.rfft(fir_coefficients_padded, axis=-1)
  log_h_k = jnp.log(jnp.abs(h_k) + 1e-7)
  ceps = jnp.fft.irfft(log_h_k, axis=-1)

  m0 = jnp.zeros((num_taps - 1,))
  m1 = jnp.ones((1,))
  m2 = jnp.ones((num_taps - 1,)) * 2.0
  lmin = jnp.concatenate([m1, m2, m1, m0])
  lmin = jnp.expand_dims(lmin, range(fir_taps.ndim - 1))

  ceps_filtered = ceps * lmin
  h_min_phase = jnp.fft.rfft(ceps_filtered, axis=-1)
  h_min_phase_exp = jnp.exp(h_min_phase)
  h_min_phase_taps = jnp.fft.irfft(h_min_phase_exp, axis=-1)

  return h_min_phase_taps[..., :num_taps]


def _resolve_num_taps(hparams, filter_len: int) -> int:
  """Resolves synthesis FIR tap count from hparams, capped at filter_len // 2."""
  num_taps = getattr(hparams, 'num_taps', filter_len // 2)
  if num_taps is None or num_taps > filter_len // 2:
    return filter_len // 2
  return num_taps


def _compute_synthesis_fir_taps(
    mask: jnp.ndarray, prototype_filter: jnp.ndarray, hparams
) -> jnp.ndarray:
  """Computes windowed (and optionally minimum-phase) FIR taps from a mask.

  Args:
    mask: Real subband gain mask of shape `[..., num_bins]`.
    prototype_filter: Prototype window of shape `[prototype_filter_length]`.
    hparams: Configuration object providing `prototype_filter_length`,
      `shift_factor`, `num_subbands`, `num_taps`, and `convert_to_min_phase`.

  Returns:
    FIR filter taps of shape `[..., new_filter_len]`.
  """
  filter_len = hparams.prototype_filter_length
  shift = hparams.shift_factor
  subbands = hparams.num_subbands

  filters = jnp.fft.irfft(mask, filter_len) * subbands / jnp.sqrt(shift)
  filters = jnp.concatenate([filters, filters], axis=-1)
  filter_index = filter_len // 2
  filters = filters[..., filter_index:-filter_index] * prototype_filter

  new_filter_len = _resolve_num_taps(hparams, filter_len)
  max_index = filter_len // 2 + 1
  start = max_index - (new_filter_len // 2)
  end = start + new_filter_len
  filters = filters[..., start:end] * hamming_window(
      new_filter_len, periodic=False
  )

  if hparams.convert_to_min_phase:
    filters = convert_fir_taps_to_min_phase_jax(filters, new_filter_len)

  return filters


def low_latency_analysis_filterbank(
    x_audio: jnp.ndarray, prototype_filter: jnp.ndarray, hparams
) -> jnp.ndarray:
  """Low latency analysis filterbank with causal left-padding."""
  filter_len = hparams.prototype_filter_length
  shift = hparams.shift_factor

  num_samples = x_audio.shape[-1]
  causal_pad = filter_len - shift
  num_frames = (num_samples + shift - 1) // shift
  tail_pad = num_frames * shift - num_samples
  x_audio_padded = jnp.pad(x_audio, ((0, 0), (causal_pad, tail_pad)))

  # Frame using convolution-like operator
  framed = jax.lax.conv_general_dilated_patches(
      x_audio_padded[..., None],
      (filter_len,),
      (shift,),
      padding='VALID',
      dimension_numbers=('NTC', 'OIT', 'NTC'),
  )  # [B, F, L]

  # Apply prototype filter
  waveforms_added = framed * prototype_filter

  # RFFT
  spectrogram = jnp.abs(jnp.fft.rfft(waveforms_added, filter_len)) + 1e-8

  # Compression
  return jnp.power(spectrogram, hparams.in_magnitude_compression)


def low_latency_synthesis_filterbank(
    x_audio: jnp.ndarray,
    prototype_filter: jnp.ndarray,
    mask: jnp.ndarray,
    hparams,
) -> jnp.ndarray:
  """Low-latency synthesis filterbank via time-domain FIR filtering (arXiv:2409.18239).

  Converts the real subband gain `mask` into a windowed FIR filter per frame
  (optionally converted to minimum phase), convolves each causally padded audio
  frame with the previous and current frame filters, and crossfades the valid
  overlap region of length `shift_factor`.

  Args:
    x_audio: Input waveform of shape `[batch, num_samples]`.
    prototype_filter: Prototype window of shape `[prototype_filter_length]`.
    mask: Subband gain mask of shape `[batch, num_frames, num_bins]`.
    hparams: Configuration object providing `prototype_filter_length`,
      `shift_factor`, `num_subbands`, `num_taps`, and `convert_to_min_phase`.

  Returns:
    Synthesized waveform of shape `[batch, num_samples]`.
  """
  filter_len = hparams.prototype_filter_length
  shift = hparams.shift_factor

  filters = _compute_synthesis_fir_taps(mask, prototype_filter, hparams)
  new_filter_len = filters.shape[-1]

  # Shifting frames for crossfading (crossfading f_{t-1} to f_t)
  filters_old = jnp.pad(filters[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
  filters_new = filters

  # FFT Conv Helper
  def fft_conv(input1, input2):
    len1 = input1.shape[-1]
    len2 = input2.shape[-1]
    fft_len = len1 + len2 - 1
    rfft1 = jnp.fft.rfft(input1, fft_len)
    rfft2 = jnp.fft.rfft(input2, fft_len)
    return jnp.fft.irfft(rfft1 * rfft2, fft_len)

  # Frame input causally to match analysis filterbank and stream_step
  num_samples = x_audio.shape[-1]
  causal_pad = filter_len - shift
  num_frames = (num_samples + shift - 1) // shift
  tail_pad = num_frames * shift - num_samples
  x_audio_padded = jnp.pad(x_audio, ((0, 0), (causal_pad, tail_pad)))

  frames = jax.lax.conv_general_dilated_patches(
      x_audio_padded[..., None],
      (filter_len,),
      (shift,),
      padding='VALID',
      dimension_numbers=('NTC', 'OIT', 'NTC'),
  )

  filtered_old = fft_conv(filters_old, frames)
  filtered_new = fft_conv(filters_new, frames)

  if hparams.convert_to_min_phase:
    complete_overlap_start = filter_len - shift
  else:
    complete_overlap_start = new_filter_len // 2 - 1

  complete_overlap_end = complete_overlap_start + shift

  # Crossfading Window
  crossfade_window = jnp.arange(1, shift + 1, dtype=jnp.float32) / shift

  crossfaded_frames = (
      filtered_old[..., complete_overlap_start:complete_overlap_end]
      * (1.0 - crossfade_window[None, None, :])
      + filtered_new[..., complete_overlap_start:complete_overlap_end]
      * crossfade_window[None, None, :]
  )

  # Overlap and add (concat since step == frame_len)
  output_waveforms = jnp.reshape(crossfaded_frames, (x_audio.shape[0], -1))

  return output_waveforms[:, :num_samples]


class VikingCell(nn.Module):
  """Submodule for a single Viking step (Parallel RNN with Hub)."""

  lstm_units: int
  hub_units: int
  num_bins: int
  num_parallel: int
  lstm_depth: int = 1
  max_mask_gain: float = 20.0

  @nn.compact
  def __call__(self, carry, x_t):
    # carry: tuple of self.lstm_depth * self.num_parallel (c, h) tuples
    # x_t: [batch, num_bins]
    features_per_lstm = self.num_bins // self.num_parallel
    new_carries = []
    branch_outputs = []
    hub = None
    concat_outputs = None

    for layer_idx in range(self.lstm_depth):
      layer_outputs = []
      for i in range(self.num_parallel):
        if layer_idx == 0:
          start = i * features_per_lstm
          end = (
              (i + 1) * features_per_lstm
              if i < self.num_parallel - 1
              else self.num_bins
          )
          branch_in = x_t[..., start:end]
        else:
          branch_in = jnp.concatenate([branch_outputs[i], hub], axis=-1)

        carry_idx = layer_idx * self.num_parallel + i
        cell_name = (
            f'lstm_{layer_idx}_{i}' if self.lstm_depth > 1 else f'lstm_{i}'
        )
        (next_c, next_h), res = nn.LSTMCell(self.lstm_units, name=cell_name)(
            carry[carry_idx], branch_in
        )
        layer_outputs.append(res)
        # Clip the cell state to prevent explosion.
        # Use min/max formulation to avoid gradient issues at boundaries.
        next_c_clipped = jnp.minimum(jnp.maximum(next_c, -10.0), 10.0)
        new_carries.append((next_c_clipped, next_h))

      branch_outputs = layer_outputs
      concat_outputs = jnp.concatenate(branch_outputs, axis=-1)
      if layer_idx < self.lstm_depth - 1:
        hub = nn.Dense(self.hub_units, name=f'hub_{layer_idx}')(concat_outputs)

    # Final hub interconnection with tanh nonlinearity
    hub = nn.tanh(nn.Dense(self.hub_units, name='hub')(concat_outputs))
    combined = jnp.concatenate([concat_outputs, hub], axis=-1)

    # Final dense for mask logits (initialized at unity gain when > 1.0)
    bias_init_val = (
        -math.log(self.max_mask_gain - 1.0) if self.max_mask_gain > 1.0 else 0.0
    )
    mask_logits = nn.Dense(
        self.num_bins,
        bias_init=nn.initializers.constant(bias_init_val),
        name='mask_dense',
    )(combined)
    mask = self.max_mask_gain * nn.sigmoid(mask_logits)

    return tuple(new_carries), mask


class VikingGenerator(nn.Module):
  """Viking Model JAX Implementation."""

  # Standard STFT Config
  frame_length: int = 256
  frame_step: int = 16

  # Low Latency Filterbank Config (Mode 3)
  use_low_latency_filterbanks: bool = False
  shift_factor: int = 64
  num_subbands: int = 512
  prototype_filter_length: int = 512
  convert_to_min_phase: bool = False
  num_taps: int = 256  # Default to new_filter_len (256) at 32kHz

  # Core Model Config
  num_parallel: int = 4
  lstm_depth: int = 1
  lstm_units: int = 32
  hub_units: int = 8
  in_magnitude_compression: float = 0.3
  max_mask_gain: float = 20.0

  @nn.compact
  def __call__(self, x_audio, train: bool = False):
    del train
    # x_audio shape: [batch, samples, 1] or [batch, samples]
    if x_audio.ndim == 3:
      x_audio = jnp.squeeze(x_audio, axis=-1)

    batch, samples = x_audio.shape

    prototype_filter = None
    stfts = None

    if self.use_low_latency_filterbanks:
      # --- Mode 3: Low Latency Filterbank ---
      prototype_filter = design_prototype_filter(self.prototype_filter_length)
      compressed_mags = low_latency_analysis_filterbank(
          x_audio, prototype_filter, self
      )
    else:
      # --- Standard Mode: STFT (causally padded to match stream_step) ---
      causal_pad = self.frame_length - self.frame_step
      num_frames = (samples + self.frame_step - 1) // self.frame_step
      tail_pad = num_frames * self.frame_step - samples
      x_audio_padded = jnp.pad(x_audio, ((0, 0), (causal_pad, tail_pad)))

      framed = jax.lax.conv_general_dilated_patches(
          x_audio_padded[..., None],
          (self.frame_length,),
          (self.frame_step,),
          padding='VALID',
          dimension_numbers=('NTC', 'OIT', 'NTC'),
      )

      # Applying window
      window = hann_window(self.frame_length).astype(x_audio.dtype)
      windowed_frames = framed * window

      stfts = jnp.fft.rfft(windowed_frames, self.frame_length)
      magnitudes = jnp.abs(stfts)
      compressed_mags = jnp.power(
          magnitudes + 1e-8, self.in_magnitude_compression
      )

    # --- Core RNN Model (Scanned) ---
    num_bins = compressed_mags.shape[-1]

    init_carry = tuple(
        (
            jnp.zeros((batch, self.lstm_units)),
            jnp.zeros((batch, self.lstm_units)),
        )
        for _ in range(self.lstm_depth * self.num_parallel)
    )

    viking_cell = VikingCell(
        lstm_units=self.lstm_units,
        hub_units=self.hub_units,
        num_bins=num_bins,
        num_parallel=self.num_parallel,
        lstm_depth=self.lstm_depth,
        max_mask_gain=self.max_mask_gain,
    )

    def body_fn(cell, carry, x):
      return cell(carry, x)

    viking_scan = nn.scan(
        body_fn,
        variable_broadcast='params',
        split_rngs={'params': False},
        in_axes=1,  # Refers to the 'x' argument
        out_axes=1,
    )

    _, masks = viking_scan(viking_cell, init_carry, compressed_mags)

    # --- Reconstruction ---
    if self.use_low_latency_filterbanks:
      assert prototype_filter is not None
      output_audio = low_latency_synthesis_filterbank(
          x_audio, prototype_filter, masks, self
      )
    else:
      denoised_stfts = masks * stfts

      # ISTFT
      ifft_frames = jnp.fft.irfft(denoised_stfts, self.frame_length)

      # Applying inverse window
      inv_window = inverse_stft_window_fn(
          self.frame_length, self.frame_step, hann_window
      )
      windowed_ifft = ifft_frames * inv_window

      output_audio = overlap_and_add(windowed_ifft, self.frame_step)

    output_audio = output_audio[:, :samples]
    return output_audio[..., None]

  def init_stream_state(self, batch_size: int = 1) -> 'VikingStreamState':
    """Initializes the stream state for chunk-by-chunk in-stream processing."""
    init_carry = tuple(
        (
            jnp.zeros((batch_size, self.lstm_units)),
            jnp.zeros((batch_size, self.lstm_units)),
        )
        for _ in range(self.lstm_depth * self.num_parallel)
    )

    if self.use_low_latency_filterbanks:
      filter_len = self.prototype_filter_length
      shift = self.shift_factor
      buf_len = filter_len - shift
      audio_buf = jnp.zeros((batch_size, buf_len), dtype=jnp.float32)
      new_filter_len = _resolve_num_taps(self, filter_len)
      filter_old = jnp.zeros((batch_size, new_filter_len), dtype=jnp.float32)
      return VikingStreamState(
          audio_buffer=audio_buf,
          lstm_carry=init_carry,
          filter_old=filter_old,
      )
    else:
      buf_len = self.frame_length - self.frame_step
      audio_buf = jnp.zeros((batch_size, buf_len), dtype=jnp.float32)
      overlap_buf = jnp.zeros(
          (batch_size, self.frame_length - self.frame_step), dtype=jnp.float32
      )
      return VikingStreamState(
          audio_buffer=audio_buf,
          lstm_carry=init_carry,
          overlap_buffer=overlap_buf,
      )

  @nn.compact
  def stream_step(
      self, stream_state: 'VikingStreamState', audio_chunk: jnp.ndarray
  ) -> tuple['VikingStreamState', jnp.ndarray]:
    """Processes a single audio chunk in-stream.

    Args:
      stream_state: Current `VikingStreamState` carrying buffered samples, LSTM
        states, and synthesis overlap/filter history.
      audio_chunk: Input audio chunk of shape `[batch, shift]` or `[batch,
        shift, 1]`, where `shift` is `self.shift_factor` when
        `self.use_low_latency_filterbanks` is True and `self.frame_step`
        otherwise.

    Returns:
      A tuple `(next_state, out_chunk)` where `out_chunk` has shape
      `[batch, shift, 1]`.

    Raises:
      ValueError: If the time dimension of `audio_chunk` does not equal `shift`.
    """
    expected_shift = (
        self.shift_factor
        if self.use_low_latency_filterbanks
        else self.frame_step
    )
    chunk_len = (
        audio_chunk.shape[1] if audio_chunk.ndim == 3 else audio_chunk.shape[-1]
    )
    if chunk_len != expected_shift:
      raise ValueError(
          f'Expected audio_chunk length {expected_shift}, got {chunk_len}'
      )

    if audio_chunk.ndim == 3:
      audio_chunk = jnp.squeeze(audio_chunk, axis=-1)

    batch = audio_chunk.shape[0]

    if self.use_low_latency_filterbanks:
      filter_len = self.prototype_filter_length
      shift = self.shift_factor
      full_frame = jnp.concatenate(
          [stream_state.audio_buffer, audio_chunk], axis=-1
      )
      next_audio_buf = full_frame[:, shift:]

      prototype_filter = design_prototype_filter(filter_len)
      waveforms_added = full_frame * prototype_filter
      spectrogram = (
          jnp.abs(jnp.fft.rfft(waveforms_added, filter_len)) + 1e-8
      ) ** self.in_magnitude_compression

      viking_cell = VikingCell(
          lstm_units=self.lstm_units,
          hub_units=self.hub_units,
          num_bins=spectrogram.shape[-1],
          num_parallel=self.num_parallel,
          lstm_depth=self.lstm_depth,
          max_mask_gain=self.max_mask_gain,
      )
      next_carry, mask = viking_cell(stream_state.lstm_carry, spectrogram)

      filter_new = _compute_synthesis_fir_taps(mask, prototype_filter, self)
      new_filter_len = filter_new.shape[-1]

      def fft_conv_1d(f, a):
        fft_len = f.shape[-1] + a.shape[-1] - 1
        return jnp.fft.irfft(
            jnp.fft.rfft(f, fft_len) * jnp.fft.rfft(a, fft_len), fft_len
        )

      assert stream_state.filter_old is not None
      filtered_old = fft_conv_1d(stream_state.filter_old, full_frame)
      filtered_new = fft_conv_1d(filter_new, full_frame)

      if self.convert_to_min_phase:
        complete_overlap_start = filter_len - shift
      else:
        complete_overlap_start = new_filter_len // 2 - 1

      complete_overlap_end = complete_overlap_start + shift
      crossfade_window = jnp.arange(1, shift + 1, dtype=jnp.float32) / shift

      out_chunk = (
          filtered_old[:, complete_overlap_start:complete_overlap_end]
          * (1.0 - crossfade_window[None, :])
          + filtered_new[:, complete_overlap_start:complete_overlap_end]
          * crossfade_window[None, :]
      )

      next_state = VikingStreamState(
          audio_buffer=next_audio_buf,
          lstm_carry=next_carry,
          filter_old=filter_new,
      )
      return next_state, out_chunk[..., None]

    else:
      frame_len = self.frame_length
      shift = self.frame_step
      full_frame = jnp.concatenate(
          [stream_state.audio_buffer, audio_chunk], axis=-1
      )
      next_audio_buf = full_frame[:, shift:]

      window = hann_window(frame_len).astype(full_frame.dtype)
      stft_frame = jnp.fft.rfft(full_frame * window, frame_len)
      compressed_mag = (
          jnp.abs(stft_frame) + 1e-8
      ) ** self.in_magnitude_compression

      viking_cell = VikingCell(
          lstm_units=self.lstm_units,
          hub_units=self.hub_units,
          num_bins=compressed_mag.shape[-1],
          num_parallel=self.num_parallel,
          lstm_depth=self.lstm_depth,
          max_mask_gain=self.max_mask_gain,
      )
      next_carry, mask = viking_cell(stream_state.lstm_carry, compressed_mag)

      denoised_stft = mask * stft_frame
      ifft_frame = jnp.fft.irfft(denoised_stft, frame_len)
      inv_window = inverse_stft_window_fn(frame_len, shift, hann_window)
      windowed_ifft = ifft_frame * inv_window

      assert stream_state.overlap_buffer is not None
      out_chunk = (
          windowed_ifft[:, :shift] + stream_state.overlap_buffer[:, :shift]
      )

      remaining_ifft = windowed_ifft[:, shift:]
      remaining_buf = stream_state.overlap_buffer[:, shift:]
      pad_zeros = jnp.zeros((batch, shift), dtype=windowed_ifft.dtype)
      next_overlap_buf = remaining_ifft + jnp.concatenate(
          [remaining_buf, pad_zeros], axis=-1
      )

      next_state = VikingStreamState(
          audio_buffer=next_audio_buf,
          lstm_carry=next_carry,
          overlap_buffer=next_overlap_buf,
      )
      return next_state, out_chunk[..., None]
