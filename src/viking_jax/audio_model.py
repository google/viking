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

"""`VikingAudioModel`: the Viking generator as an `hp_acoustic` `AudioModel`."""

from hp_acoustic.models import audio_model as audio_model_lib

from viking_jax import viking_jax

# Register VikingModel factory with hp_acoustic audio_model registry.
audio_model_lib.register_model('viking', viking_jax.VikingModel.from_config)

VikingModel = viking_jax.VikingModel
VikingAudioModel = viking_jax.VikingModel
VikingParams = viking_jax.VikingParams
VikingGenerator = viking_jax.VikingGenerator

__all__ = [
  'VikingModel',
  'VikingAudioModel',
  'VikingParams',
  'VikingGenerator',
]
