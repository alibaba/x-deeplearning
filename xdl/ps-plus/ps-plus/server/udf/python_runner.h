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

#ifndef PS_PLUS_COMMON_PYTHON_RUNNER_H_
#define PS_PLUS_COMMON_PYTHON_RUNNER_H_

#include <vector>
#include <string>
#include "Python.h"
#include "ps-plus/common/types.h"
#include "ps-plus/common/tensor_shape.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/status.h"

namespace ps {

class PythonRunner {
 public:
  struct NumpyArray {
    ~NumpyArray();
    NumpyArray();
    NumpyArray(
        void* data, DataType type, const TensorShape& shape, PyObject* object);
    NumpyArray(const NumpyArray&);
    NumpyArray& operator=(const NumpyArray&);
    void* data;
    DataType type;
    TensorShape shape;
    PyObject* object;
  };
  
  PythonRunner() : func_(nullptr) {}
  ~PythonRunner() {Py_XDECREF(func_);}

  Status Init(const std::string& func_def, const std::string& func_name);

  Status Run(const std::vector<NumpyArray>& inputs);
  Status Run(const std::vector<NumpyArray>& inputs, NumpyArray* result);
  Status Run(const std::vector<NumpyArray>& inputs, NumpyArray* r1, NumpyArray* r2);
  static Status ParseSubTensor(Tensor t, size_t seg_id, size_t max_id, NumpyArray* arr);
  static Status ParseTensor(Tensor t, NumpyArray* arr);
 private:
  Status RunImpl(const std::vector<NumpyArray>& inputs, PyObject** result);
  Status ParseObject(PyObject* arr, NumpyArray* result);
  PyObject* func_;
};


}  // namespace ps

#endif  // PS_PLUS_COMMON_PYTHON_RUNNER_H_

