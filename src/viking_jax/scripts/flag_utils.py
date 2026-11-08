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

"""Shared flag helpers for Viking DAL CLI scripts."""


def positive_or_none(count: int | None) -> int | None:
  """Maps `<= 0` ('all examples' / 'no limit') and `None` onto `None`."""
  return count if (count is not None and count > 0) else None
