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
from xdl.python.lib.datatype import DataType
from xdl.python.lib.tensorshape import TensorShape
from xdl.python.lib.graph import create_op
from xdl.python.lib.tensor import Tensor, convert_to_tensor
from xdl.python.lib.error import check_error
import ctypes
import contextlib
import inspect
import types

class PyBuilder(object):
  indent = None
  stmts = None

  def __init__(self):
    self.indent = ""
    self.stmts = []

  def append_stmt(self, fmt, *args):
    self.stmts.append(self.indent + fmt.format(*args) + "\n")

  @contextlib.contextmanager
  def indent_stmt(self, fmt, *args):
    self.stmts.append(self.indent + fmt.format(*args) + "\n")
    try:
      old_indent = self.indent
      self.indent = self.indent + "  "
      yield
    finally:
      self.indent = old_indent

  def get_stmts(self):
    return "".join(self.stmts)

  def get_numbered_stmts(self):
    indent = 0
    stmts = []
    for i in range(1, len(self.stmts) + 1):
      if i >= 10 ** indent:
        indent += 1
      stmts.append(" " * (5 - indent) + str(i) + "| " + self.stmts[i - 1])
    return "".join(stmts)

def _generate_check_set_attr(builder, attr, value):
  with builder.indent_stmt("if {0} is None:", attr):
    builder.append_stmt("{0} = {1}", attr, value)
  with builder.indent_stmt("elif {0} != {1}:", attr, value):
    builder.append_stmt("raise ValueError('attr check error on {0}')", attr)

def _generate_default_attr(builder, attr, value):
  with builder.indent_stmt("if {0} is None:", attr):
    builder.append_stmt("{0} = {1}", attr, value)

def _generate_check_attr(builder, attr, t):
  with builder.indent_stmt("if not isinstance({0}, {1}):", attr, t):
    builder.append_stmt("raise ValueError('Attr {0} should be {1} but is ' + repr({0}))", attr, t)

def _create_func_name(name):
  ret = []
  add_slash = False;
  for c in name:
    if 'A' <= c and c <= 'Z':
      if add_slash:
        ret.append('_')
      ret.append(c.lower())
      add_slash = False
    elif 'a' <= c and c <= 'z':
      ret.append(c)
      add_slash = True
    else:
      ret.append(c)
      add_slash = False
  return ''.join(ret)

_func_code = {}
_op_def = {}

def op_generate(opdef):
  check_error(opdef.status)
  func_name = _create_func_name(opdef.name)
  inputs = opdef.inputs
  outputs = opdef.outputs
  attrs = opdef.attrs
  derivate_attrs = set()
  default_attrs = set()
  for i in inputs:
    if i.type.repeated == pybind.OpDefineItem.RepeatType.no_repeat:
      if i.type.type.attr != "":
        derivate_attrs.add(i.type.type.attr)
    elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_and_size:
      if i.type.type.attr != "":
        derivate_attrs.add(i.type.type.attr)
      if i.type.size.attr != "":
        derivate_attrs.add(i.type.size.attr)
    elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_list:
      derivate_attrs.add(i.type.type_list)

  for i in attrs:
    if i.default_value.attr_type != pybind.AttrValue.Type.none:
      default_attrs.add(i.name)

  func_input = []
  func_output = []
  for i in inputs:
    func_input.append(i.name)
  for i in attrs:
    if i.name not in derivate_attrs:
      if i.name not in default_attrs:
        func_input.append(i.name)
  for i in attrs:
    if i.name in derivate_attrs or i.name in default_attrs:
      func_input.append(i.name + " = None")
  for i in outputs:
    func_output.append(i.name)

  builder = PyBuilder()

  # def
  if len(func_input) == 0:
    def_fmt = "def {0}({1}name_ = None):"
  else:
    def_fmt = "def {0}({1}, name_ = None):"
  with builder.indent_stmt(def_fmt, func_name, ', '.join(func_input)):
    with builder.indent_stmt("try:"):
      # convert to tensor
      for i in inputs:
        if i.type.repeated == pybind.OpDefineItem.RepeatType.no_repeat:
          builder.append_stmt("{0} = convert_to_tensor({0})", i.name)
        elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_and_size:
          builder.append_stmt("{0} = [convert_to_tensor(i) for i in {0}]", i.name)
        elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_list:
          builder.append_stmt("{0} = [convert_to_tensor(i) for i in {0}]", i.name)
      builder.append_stmt("")

      # derivate_attrs
      for i in inputs:
        if i.type.repeated == pybind.OpDefineItem.RepeatType.no_repeat:
          if i.type.type.attr != "":
            _generate_check_set_attr(builder, i.type.type.attr, i.name + ".dtype")
            builder.append_stmt("")
        elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_and_size:
          if i.type.type.attr != "":
            with builder.indent_stmt("for i in {0}:", i.name):
              _generate_check_set_attr(builder, i.type.type.attr, "i.dtype")
            builder.append_stmt("")
          if i.type.size.attr != "":
            _generate_check_set_attr(builder, i.type.size.attr, "len({0})".format(i.name))
            builder.append_stmt("")
        elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_list:
          with builder.indent_stmt("if {0} is None:", i.type.type_list):
            builder.append_stmt("{0} = []", i.type.type_list)
            with builder.indent_stmt("for i in {0}:", i.name):
              builder.append_stmt("{0}.append(i.dtype)", i.type.type_list)
          builder.append_stmt("")

      # append default attr
      for i in attrs:
        if i.default_value.attr_type == pybind.AttrValue.Type.none:
          continue
        elif i.default_value.attr_type == pybind.AttrValue.Type.string:
          _generate_default_attr(builder, i.name, repr(i.default_value.s))
        elif i.default_value.attr_type == pybind.AttrValue.Type.int:
          _generate_default_attr(builder, i.name, repr(i.default_value.i))
        elif i.default_value.attr_type == pybind.AttrValue.Type.float:
          _generate_default_attr(builder, i.name, repr(i.default_value.f))
        elif i.default_value.attr_type == pybind.AttrValue.Type.bool:
          _generate_default_attr(builder, i.name, repr(i.default_value.b))
        elif i.default_value.attr_type == pybind.AttrValue.Type.shape:
          _generate_default_attr(builder, i.name, repr(TensorShape(i.default_value.shape)))
        elif i.default_value.attr_type == pybind.AttrValue.Type.type:
          _generate_default_attr(builder, i.name, repr(DataType(i.default_value.type)))
        else:
          _generate_default_attr(builder, i.name, repr(i.default_value.type_list))
        builder.append_stmt("")

      # check attr
      for i in attrs:
        if i.type == pybind.AttrValue.Type.none:
          raise ValueError("Attr Type should not be none")
        elif i.type == pybind.AttrValue.Type.string:
          _generate_check_attr(builder, i.name, "(str, unicode, bytes)")
        elif i.type == pybind.AttrValue.Type.int:
          _generate_check_attr(builder, i.name, "(int, long)")
        elif i.type == pybind.AttrValue.Type.float:
          _generate_check_attr(builder, i.name, "(float)")
        elif i.type == pybind.AttrValue.Type.bool:
          _generate_check_attr(builder, i.name, "(bool)")
        elif i.type == pybind.AttrValue.Type.shape:
          _generate_check_attr(builder, i.name, "(TensorShape, list, tuple)")
        elif i.type == pybind.AttrValue.Type.type:
          _generate_check_attr(builder, i.name, "(DataType)")
        elif i.type == pybind.AttrValue.Type.type_list:
          _generate_check_attr(builder, i.name, "(list, tuple)")
        else:
          raise NotImplementedError
        builder.append_stmt("")

      # check input dtype
      for i in inputs:
        if i.type.type.attr != "":
          dtype = i.type.type.attr
        else:
          dtype = repr(DataType(i.type.type.raw))
        if i.type.size.attr != "":
          dsize = i.type.size.attr
        else:
          dsize = repr(i.type.size.raw)
        if i.type.repeated == pybind.OpDefineItem.RepeatType.no_repeat:
          with builder.indent_stmt("if {0}.dtype != {1}:", i.name, dtype):
            builder.append_stmt("raise ValueError('Check Input {0} type should be' + repr({1}) + ' but is ' + repr({0}.dtype))", i.name, dtype)
        elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_and_size:
          with builder.indent_stmt("if len({0}) != {1}:", i.name, dsize):
            builder.append_stmt("raise ValueError('Check Input {0} size should be ' + repr({1}) + ' but is ' + repr(len({0})))", i.name, dsize)
          with builder.indent_stmt("for i in range({0}):", dsize):
            with builder.indent_stmt("if {0}[i].dtype != {1}:", i.name, dtype):
              builder.append_stmt("raise ValueError('Check Input {0} type should be ' + repr({1}) + ' but is ' + repr({0}[i].dtype))", i.name, dtype)
        elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_list:
          with builder.indent_stmt("if len({0}) != len({1}):", i.name, i.type.type_list):
            builder.append_stmt("raise ValueError('Check Input {0} size should be ' + repr(len({1})) + ' but is ' + repr(len({0})))", i.name, i.type.type_list)
          with builder.indent_stmt("for i in range(len({0})):", i.name):
            with builder.indent_stmt("if {0}[i].dtype != {1}[i]:", i.name, i.type.type_list):
              builder.append_stmt("raise ValueError('Check Input {0}[' + repr(i) + '] type should be ' + repr({1}[i]) + ' but is ' + repr({0}[i].dtype))", i.name, i.type.type_list)
        else:
          raise NotImplementedError
        builder.append_stmt("")

      # prepare CreateOp
      builder.append_stmt("__input = [{0}]", ", ".join([i.name for i in inputs]))
      builder.append_stmt("__args = {0}{2}{1}", "{", "}", ", ".join(["'{0}' : {0}".format(i.name) for i in attrs]))
      output_spec = []
      for i in outputs:
        if i.type.type.attr != "":
          dtype = i.type.type.attr
        else:
          dtype = repr(DataType(i.type.type.raw))
        if i.type.size.attr != "":
          dsize = i.type.size.attr
        else:
          dsize = repr(i.type.size.raw)
        if i.type.repeated == pybind.OpDefineItem.RepeatType.no_repeat:
          output_spec.append("{0}".format(dtype))
        elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_and_size:
          output_spec.append("[{0}] * {1}".format(dtype, dsize))
        elif i.type.repeated == pybind.OpDefineItem.RepeatType.type_list:
          output_spec.append(i.type.type_list)
      builder.append_stmt("__output_spec = [{0}]", ", ".join(output_spec))
      builder.append_stmt("")

      builder.append_stmt("__op = create_op(_op_def['{0}'], name_, __input, __args, __output_spec)", opdef.name)
      builder.append_stmt("")

      if len(outputs) == 0:
        builder.append_stmt("return __op.depend()")
      elif len(outputs) == 1:
        builder.append_stmt("return __op.outputs[0]")
      else:
        builder.append_stmt("return __op.outputs")
    with builder.indent_stmt("except Exception as e:"):
      builder.append_stmt("print")
      builder.append_stmt("print _func_code['{0}']", opdef.name)
      builder.append_stmt("raise")

  builder.append_stmt("result_func = {0}", func_name)

  exec(builder.get_stmts())
  _func_code[opdef.name] = builder.get_numbered_stmts()
  _op_def[opdef.name] = opdef
  result_func.code_text = _func_code[opdef.name]
  return func_name, result_func

def generate_module(name):
  ret = types.ModuleType(name)
  for opdef in pybind.GetLatestOpDefineItem():
    name, op = op_generate(opdef)
    setattr(ret, name, op)
  return ret

def load_op_library(file_name):
  module_name = file_name.split('/')[-1]
  module_name = module_name.split('.')[0]
  if module_name[:3] == 'lib':
    module_name = module_name[3:]
  ctypes.CDLL(file_name, ctypes.RTLD_GLOBAL)
  return generate_module(module_name)
