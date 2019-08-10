# Copyright 2018 Alibaba Group. All Rights Reserved.
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
# ==============================================================================

import contextlib

_ROOT_SCOPE = ''
_CUR_SCOPE = _ROOT_SCOPE
_SCOPE_SET = set([_ROOT_SCOPE])
  
@contextlib.contextmanager
def model_scope(scope):
  global _SCOPE_SET
  if scope not in _SCOPE_SET:
    _SCOPE_SET.add(scope)
  global _CUR_SCOPE
  try:
    old_scope = _CUR_SCOPE
    _CUR_SCOPE = scope
    yield
  finally:
    _CUR_SCOPE = old_scope

def cur_model_scope():
  global _CUR_SCOPE
  return _CUR_SCOPE

def get_model_scopes():
  global _SCOPE_SET
  return _SCOPE_SET

