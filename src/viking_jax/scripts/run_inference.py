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

"""Runs single-WAV inference using a trained Viking DAL checkpoint.

Supports both full-sequence inference (`--streaming=false`) and real-time
hop-by-hop stateful streaming (`--streaming=true`, via `VikingGenerator`'s
`init_stream_state` and `stream_step`).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from absl import flags
from hp_acoustic import audio_constants
from hp_acoustic.evaluation import report
from hp_acoustic.io import audio_io

from viking_jax import audio_model as _viking_audio_model  # noqa: F401
from viking_jax import viking_jax

_MODEL_CHECKPOINT_PATH = flags.DEFINE_string(
  'model_checkpoint_path',
  None,
  'Path to a saved Viking model checkpoint.',
  required=True,
)
_INPUT_WAV_PATH = flags.DEFINE_string(
  'input_wav_path',
  None,
  'Path to the input WAV file.',
  required=True,
)
_OUTPUT_WAV_PATH = flags.DEFINE_string(
  'output_wav_path',
  None,
  'Path to save the enhanced output WAV file.',
  required=True,
)
_STREAMING = flags.DEFINE_boolean(
  'streaming',
  True,
  'Run chunk-by-chunk stateful streaming inference via `stream_step`.',
)
_TARGET_SAMPLE_RATE = flags.DEFINE_integer(
  'target_sample_rate',
  audio_constants.DEFAULT_SAMPLE_RATE,
  'Target sample rate in Hz.',
)


def _run_viking_streaming_wav(
  module: viking_jax.VikingGenerator,
  params: viking_jax.VikingParams,
  input_audio: np.ndarray,
) -> np.ndarray:
  """Runs hop-by-hop stateful streaming inference over `input_audio`."""
  hop = (
    module.shift_factor
    if module.use_low_latency_filterbanks
    else module.frame_step
  )
  original_len = len(input_audio)
  pad_len = (-original_len) % hop
  padded = (
    np.pad(input_audio, (0, pad_len), mode='constant')
    if pad_len
    else input_audio
  )

  state = module.init_stream_state(batch_size=1)

  @jax.jit
  def step_fn(s, chunk):
    return module.apply({'params': params}, s, chunk, method=module.stream_step)

  chunks = []
  for start in range(0, len(padded), hop):
    chunk = jnp.asarray(padded[start : start + hop][None, :, None])
    state, out_chunk = step_fn(state, chunk)
    chunks.append(np.asarray(jnp.squeeze(out_chunk, axis=(0, -1))))

  return np.concatenate(chunks)[:original_len]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  input_audio, _ = audio_io.read_wav(
    _INPUT_WAV_PATH.value, _TARGET_SAMPLE_RATE.value
  )
  generator_model, params_g = report.restore_generator(
    _MODEL_CHECKPOINT_PATH.value,
    model=viking_jax.VikingModel(),
  )

  if len(input_audio) == 0:
    predicted_audio = np.zeros(0, dtype=np.float32)
  elif _STREAMING.value and isinstance(generator_model, viking_jax.VikingModel):
    predicted_audio = _run_viking_streaming_wav(
      generator_model.module, params_g, np.asarray(input_audio)
    )
  else:
    input_bcl = jnp.asarray(input_audio[np.newaxis, :, np.newaxis])
    out_bcl = generator_model.apply(params_g, input_bcl)
    predicted_audio = np.atleast_1d(np.asarray(jnp.squeeze(out_bcl)))

  audio_io.save_audio_to_file(
    predicted_audio, _OUTPUT_WAV_PATH.value, _TARGET_SAMPLE_RATE.value
  )


if __name__ == '__main__':
  app.run(main)
