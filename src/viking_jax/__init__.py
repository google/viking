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

"""Viking JAX package."""

from viking_jax import viking_jax

VikingStreamState = viking_jax.VikingStreamState
VikingCell = viking_jax.VikingCell
VikingGenerator = viking_jax.VikingGenerator
VikingModel = viking_jax.VikingModel
VikingAudioModel = viking_jax.VikingAudioModel
VikingParams = viking_jax.VikingParams
hann_window = viking_jax.hann_window
hamming_window = viking_jax.hamming_window
design_prototype_filter = viking_jax.design_prototype_filter
convert_fir_taps_to_min_phase_jax = viking_jax.convert_fir_taps_to_min_phase_jax
low_latency_analysis_filterbank = viking_jax.low_latency_analysis_filterbank
low_latency_synthesis_filterbank = viking_jax.low_latency_synthesis_filterbank

try:
  from hp_acoustic.models import audio_model as _audio_model_lib
except ImportError:
  pass
else:
  _audio_model_lib.register_model('viking', VikingModel.from_config)

__all__ = [
  "VikingStreamState",
  "VikingCell",
  "VikingGenerator",
  "VikingModel",
  "VikingAudioModel",
  "VikingParams",
  "hann_window",
  "hamming_window",
  "design_prototype_filter",
  "convert_fir_taps_to_min_phase_jax",
  "low_latency_analysis_filterbank",
  "low_latency_synthesis_filterbank",
]
