/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "python_runner.h"
#include <mutex>
#include "numpy/npy_common.h"
#include "numpy/ndarrayobject.h"

namespace ps {

namespace {
int DtypeToNumpy(DataType type) {
  switch(type) {
  case DataType::kInt8: return NPY_INT8;
  case DataType::kInt16: return NPY_INT16;
  case DataType::kInt32: return NPY_INT32;
  case DataType::kInt64: return NPY_INT64;
  case DataType::kFloat: return NPY_FLOAT;
  case DataType::kDouble: return NPY_DOUBLE;
  default: return -1;
  };
}

Status NumpyToDtype(int x, DataType* type) {
  switch(x) {
    case NPY_BOOL: case NPY_INT8: *type = DataType::kInt8; return Status::Ok();
    case NPY_INT16: *type = DataType::kInt16; return Status::Ok();
    case NPY_INT32: *type = DataType::kInt32; return Status::Ok();
    case NPY_INT64: *type = DataType::kInt64; return Status::Ok();
    case NPY_FLOAT: *type = DataType::kFloat; return Status::Ok();
    case NPY_DOUBLE: *type = DataType::kDouble; return Status::Ok();
    default: return Status::ArgumentError("return type error " + std::to_string(x));
  };
}

class PythonContext {
 public:
  PythonContext() : lock_(mu_) {
    if (!inited_) {
      Py_Initialize();
      PyRun_SimpleString("import traceback");
      PyRun_SimpleString("import numpy");
      import_array();
      PyObject* mainModule = PyImport_ImportModule("__main__" );
      PyObject* dict = PyModule_GetDict(mainModule);
      PyObject* traceback = PyDict_GetItemString(dict, "traceback");
      format_exception = PyObject_GetAttrString(traceback, "format_exception");
      format_exception_only = PyObject_GetAttrString(traceback, "format_exception_only");
      if (format_exception == nullptr || format_exception_only == nullptr) {
        std::cerr << "traceback module import error." << std::endl;
        abort();
      }
      inited_ = true;
    }
    PyErr_Clear();
  }
  std::string getException() {
    PyObject *ptype = nullptr, *pvalue = nullptr, *ptraceback = nullptr, *pstr;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if (ptype == nullptr) {
      return "Unknown Python Exception";
    }
    if (ptraceback != nullptr) {
      pstr = PyObject_CallFunction(format_exception, "OOO", ptype, pvalue, ptraceback);
    } else {
      pstr = PyObject_CallFunction(format_exception_only, "OO", ptype, pvalue);
    }
    PyObject* slash_n = PyString_FromString("\n");
    pstr = PyObject_CallMethod(slash_n, "join", "O", pstr);
    char *pStrErrorMessage = PyString_AsString(pstr);
    PyErr_Clear();
    return pStrErrorMessage;
  }
 private:
  std::unique_lock<std::mutex> lock_;
  static PyObject *format_exception, *format_exception_only;
  static bool inited_;
  static std::mutex mu_;
};

bool PythonContext::inited_ = false;
std::mutex PythonContext::mu_;
PyObject *PythonContext::format_exception, *PythonContext::format_exception_only;
}

PythonRunner::NumpyArray::~NumpyArray() {
  Py_XDECREF(object);
}

PythonRunner::NumpyArray::NumpyArray() {
  this->object = nullptr;
}

PythonRunner::NumpyArray::NumpyArray(
    void* data, DataType type, const TensorShape& shape, PyObject* object) {
  this->data = data;
  this->type = type;
  this->shape = shape;
  this->object = object;
}

PythonRunner::NumpyArray::NumpyArray(const PythonRunner::NumpyArray& rhs) {
  this->data = rhs.data;
  this->type = rhs.type;
  this->shape = rhs.shape;
  this->object = rhs.object;
  Py_XINCREF(object);
}

PythonRunner::NumpyArray& PythonRunner::NumpyArray::operator=(const PythonRunner::NumpyArray& rhs) {
  Py_XDECREF(this->object);
  this->data = rhs.data;
  this->type = rhs.type;
  this->shape = rhs.shape;
  this->object = rhs.object;
  Py_XINCREF(object);
}

Status PythonRunner::Init(
    const std::string& func_def,
    const std::string& func_name) {
  PythonContext ctx;
  PyRun_SimpleString(func_def.c_str());
  if (PyErr_Occurred()) {
    return Status::ArgumentError("Cannot Create Python Func\n" + ctx.getException());
  }
  PyObject* mainModule = PyImport_ImportModule("__main__" );
  PyObject* dict = PyModule_GetDict(mainModule);
  func_ = PyDict_GetItemString(dict, func_name.c_str());
  if (func_ == nullptr) {
    return Status::ArgumentError("Python Func Name Error\n" + ctx.getException());
  }
  Py_XINCREF(func_);
  return Status::Ok();
}

Status PythonRunner::RunImpl(
    const std::vector<PythonRunner::NumpyArray>& inputs,
    PyObject** result) {
  PythonContext ctx;
  PyObject* result_object;
  switch (inputs.size()) {
  case 0:result_object = PyObject_CallFunction(func_, ""); break;
  case 1:result_object = PyObject_CallFunction(func_, "O", inputs[0].object); break;
  case 2:result_object = PyObject_CallFunction(func_, "OO", inputs[0].object, inputs[1].object); break;
  case 3:result_object = PyObject_CallFunction(func_, "OOO", inputs[0].object, inputs[1].object, inputs[2].object); break;
  case 4:result_object = PyObject_CallFunction(func_, "OOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object); break;
  case 5:result_object = PyObject_CallFunction(func_, "OOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object); break;
  case 6:result_object = PyObject_CallFunction(func_, "OOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object); break;
  case 7:result_object = PyObject_CallFunction(func_, "OOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object); break;
  case 8:result_object = PyObject_CallFunction(func_, "OOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object); break;
  case 9:result_object = PyObject_CallFunction(func_, "OOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object); break;
  case 10:result_object = PyObject_CallFunction(func_, "OOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object); break;
  case 11:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object); break;
  case 12:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object); break;
  case 13:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object); break;
  case 14:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object); break;
  case 15:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object); break;
  case 16:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object); break;
  case 17:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object); break;
  case 18:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object); break;
  case 19:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object); break;
  case 20:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object); break;
  case 21:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object); break;
  case 22:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object); break;
  case 23:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object); break;
  case 24:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object, inputs[23].object); break;
  case 25:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object, inputs[23].object, inputs[24].object); break;
  case 26:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object, inputs[23].object, inputs[24].object, inputs[25].object); break;
  case 27:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object, inputs[23].object, inputs[24].object, inputs[25].object, inputs[26].object); break;
  case 28:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object, inputs[23].object, inputs[24].object, inputs[25].object, inputs[26].object, inputs[27].object); break;
  case 29:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object, inputs[23].object, inputs[24].object, inputs[25].object, inputs[26].object, inputs[27].object, inputs[28].object); break;
  case 30:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object, inputs[23].object, inputs[24].object, inputs[25].object, inputs[26].object, inputs[27].object, inputs[28].object, inputs[29].object); break;
  case 31:result_object = PyObject_CallFunction(func_, "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO", inputs[0].object, inputs[1].object, inputs[2].object, inputs[3].object, inputs[4].object, inputs[5].object, inputs[6].object, inputs[7].object, inputs[8].object, inputs[9].object, inputs[10].object, inputs[11].object, inputs[12].object, inputs[13].object, inputs[14].object, inputs[15].object, inputs[16].object, inputs[17].object, inputs[18].object, inputs[19].object, inputs[20].object, inputs[21].object, inputs[22].object, inputs[23].object, inputs[24].object, inputs[25].object, inputs[26].object, inputs[27].object, inputs[28].object, inputs[29].object, inputs[30].object); break;
  default: return Status::ArgumentError("Python Call should less than 31 arguments");
  }
  if (PyErr_Occurred()) {
    return Status::ArgumentError("Python Call run with some error\n" + ctx.getException());
  }
  if (result_object == nullptr) {
    return Status::ArgumentError("Python Call run return None");
  }
  *result = result_object;
  return Status::Ok();
}
Status PythonRunner::ParseObject(PyObject* arr, PythonRunner::NumpyArray* result) {
  PyArrayObject* obj = (PyArrayObject*)PyArray_FROM_OF(arr, NPY_ARRAY_IN_ARRAY);
  if (obj == nullptr) {
    return Status::ArgumentError("Cannot convert to numpy array");
  }
  DataType dtype;
  Status conv_status = NumpyToDtype(PyArray_TYPE(obj), &dtype);
  if (!conv_status.IsOk()) {
    Py_XDECREF(obj);
    return conv_status;
  }
  result->object = (PyObject*)obj;
  result->data = PyArray_DATA(obj);
  int dim_size = PyArray_NDIM(obj);
  npy_intp * dims = PyArray_DIMS(obj);
  result->shape = TensorShape(std::vector<size_t>(dims, dims+dim_size));
  result->type = dtype;
  return Status::Ok();
}

Status PythonRunner::Run(const std::vector<PythonRunner::NumpyArray>& inputs) {
  PyObject* obj;
  PS_CHECK_STATUS(RunImpl(inputs, &obj));
  return Status::Ok();
}
Status PythonRunner::Run(
    const std::vector<PythonRunner::NumpyArray>& inputs,
    PythonRunner::NumpyArray* result) {
  PyObject* obj;
  PS_CHECK_STATUS(RunImpl(inputs, &obj));
  PythonContext ctx;
  Status st = ParseObject(obj, result);
  Py_XDECREF(obj);
  return st;
}

Status PythonRunner::Run(
    const std::vector<PythonRunner::NumpyArray>& inputs,
    PythonRunner::NumpyArray* r1,
    PythonRunner::NumpyArray* r2) {
  PyObject* obj, *obj1, *obj2;
  PS_CHECK_STATUS(RunImpl(inputs, &obj));
  PythonContext ctx;
  if (!PyArg_ParseTuple(obj, "OO", &obj1, &obj2)) {
    Py_XDECREF(obj);
    return Status::ArgumentError("Parse Result Error!\n" + ctx.getException());
  }
  Status st1 = ParseObject(obj1, r1);
  Status st2 = ParseObject(obj2, r2);
  Py_XDECREF(obj);
  if (!st1.IsOk()) {
    return st1;
  } else {
    return st2;
  }
}

Status PythonRunner::ParseSubTensor(
    Tensor t, size_t seg_id, size_t max_id, PythonRunner::NumpyArray* arr) {
  size_t segment_size = t.SegmentSize();
  void* data = t.Raw<void>(segment_size * seg_id);
  arr->shape = t.Shape();
  arr->shape.Set(0, std::min(segment_size, max_id - seg_id * segment_size));
  arr->data = data;
  arr->type = t.Type();
  std::vector<npy_intp> shapex;
  for (int i = 0; i < arr->shape.Size(); i++) {
    shapex.push_back(arr->shape[i]);
  }
  int type = DtypeToNumpy(arr->type);
  if (type == -1) {
    return Status::ArgumentError("input Type Error");
  }
  int ndim = shapex.size();
  arr->object = PyArray_SimpleNewFromData(
      ndim, &shapex[0], type, data);
  return Status::Ok();
}

Status PythonRunner::ParseTensor(
    Tensor t, PythonRunner::NumpyArray* arr) {
  arr->shape = t.Shape();
  arr->data = t.Raw<void>();
  arr->type = t.Type();
  std::vector<npy_intp> shapex;
  for (int i = 0; i < arr->shape.Size(); i++) {
    shapex.push_back(arr->shape[i]);
  }
  int type = DtypeToNumpy(arr->type);
  if (type == -1) {
    return Status::ArgumentError("input Type Error");
  }
  int ndim = shapex.size();
  arr->object = PyArray_SimpleNewFromData(
      ndim, &shapex[0], type, arr->data);
  return Status::Ok();
}

}  // namespace ps

