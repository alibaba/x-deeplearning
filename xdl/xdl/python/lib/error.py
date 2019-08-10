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

from xdl.python import pybind

OK = pybind.Status.ErrorCode.OK
ARGUMENT_ERROR = pybind.Status.ErrorCode.ArgumentError
INDEX_OVERFLOW = pybind.Status.ErrorCode.IndexOverflow
PS_ERROR = pybind.Status.ErrorCode.PsError
INTERNAL_ERROR = pybind.Status.ErrorCode.Internal
OUT_OF_RANGE = pybind.Status.ErrorCode.OutOfRange
REACH_END = pybind.Status.ErrorCode.ReachEnd

def check_error(status):
  if status.code == OK:
    return
  elif status.code == ARGUMENT_ERROR:
    raise ArgumentError(status.msg)
  elif status.code == INDEX_OVERFLOW:
    raise IndexOverflow(status.msg)
  elif status.code == PS_ERROR:
    raise PsError(status.msg)
  elif status.code == INTERNAL_ERROR:
    raise InternalError(status.msg)
  elif status.code == OUT_OF_RANGE:
    raise OutOfRange(status.msg)
  elif status.code == REACH_END:
    raise ReachEnd(status.msg)
  else:
    raise UnhandledError(status.code, status.msg)

class XdlException(Exception):
  def __init__(self):
    self._code = None
    self._msg = None
  def __str__(self):
    return "ErrorCode[%d], ErrorMsg[%s]" % (self._code, self._msg)
  @property
  def msg(self):
    return self._msg
  @property
  def code(self):
    return self._code

class ArgumentError(XdlException):
  def __init__(self, msg):
    self._code = ARGUMENT_ERROR
    self._msg = msg

class IndexOverflow(XdlException):
  def __init__(self, msg):
    self._code = INDEX_OVERFLOW
    self._msg = msg

class PsError(XdlException):
  def __init__(self, msg):
    self._code = PS_ERROR
    self._msg = msg

class InternalError(XdlException):
  def __init__(self, msg):
    self._code = INTERNAL_ERROR
    self._msg = msg

class OutOfRange(XdlException):
  def __init__(self, msg):
    self._code = OUT_OF_RANGE
    self._msg = msg

class ReachEnd(XdlException):
  def __init__(self, msg):
    self._code = REACH_END
    self._msg = msg

class UnhandledError(XdlException):
  def __init__(self, code, msg):
    self._code = code
    self._msg = msg

