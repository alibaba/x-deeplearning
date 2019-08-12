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

#include "xdl/python/pybind/executor_wrapper.h"

#include <future>
#include "xdl/core/utils/logging.h"

#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "xdl/core/framework/graph_def.h"
#include "xdl/core/framework/executor.h"

#define ONE_ARG(...) __VA_ARGS__
PYBIND11_MAKE_OPAQUE(ONE_ARG(std::unordered_map<std::string, std::string>));
PYBIND11_MAKE_OPAQUE(ONE_ARG(std::unordered_map<std::string, xdl::AttrValue>));
PYBIND11_MAKE_OPAQUE(std::vector<xdl::DataType>);
PYBIND11_MAKE_OPAQUE(std::vector<xdl::NodeDef>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<xdl::Tensor>);
PYBIND11_MAKE_OPAQUE(std::vector<xdl::OutputSpec>);
PYBIND11_MAKE_OPAQUE(std::vector<size_t>);

namespace xdl {
namespace python_lib {

struct ExecutorInstance : public Singleton<ExecutorInstance> {
 public:
  ExecutorInstance() : executor_(new Executor(ThreadPool::Global())) {}
  Executor* executor() {
    return executor_.get();
  }
 private:
  std::unique_ptr<Executor> executor_;
};

struct RunStatistic {
  std::string perf_result;
};

struct ExecuteResult {
  Status status;
  std::vector<Tensor> outputs;
  RunStatistic run_statistic;
};

struct ExecuteLoopResult : public Singleton<ExecuteLoopResult> {
  ExecuteLoopResult() : status_set(false) {}
  std::mutex mu;
  bool status_set;
  std::unique_ptr<std::promise<Status>> status;
};

struct ExecuteLoopSpec {
  GraphDef def;
  std::vector<OutputSpec> outputs;
  int id;
};

ExecuteResult Execute(const GraphDef& def, 
                      const OutputSpec& output,
                      const RunOption& run_option) {
  static Executor executor(ThreadPool::Global());
  ExecuteResult ret;
  std::promise<int> result;
  ExecutorInstance::Instance()->executor()->Run(
      def, output, run_option,
      [&](Status st, const std::vector<Tensor>& outputs, 
          const std::unordered_map<std::string, Any>& extra_info) {
    ret.status = st;
    ret.outputs = outputs;
    if (run_option.perf) {
      auto it = extra_info.find("PERF_RESULT");
      if (it != extra_info.end()) {
        ret.run_statistic.perf_result = it->second.AnyCast<std::string>();
      }
    }

    result.set_value(1);
  });
  result.get_future().wait();
  return ret;
}

void ExecuteLoopImpl(ExecuteLoopSpec* spec) {
  static Executor executor(ThreadPool::Global());
  RunOption run_option;
  ExecutorInstance::Instance()->executor()->Run(spec->def, spec->outputs[spec->id], run_option,
  [spec](Status st, const std::vector<Tensor>& outputs,
	      const std::unordered_map<std::string, Any>& extra_info) {
    if (!st.IsOk()) {
      ExecuteLoopResult* result = ExecuteLoopResult::Instance();
      std::unique_lock<std::mutex> lock(result->mu);
      if (!result->status_set) {
        if (result->status == nullptr) {
          result->status.reset(new std::promise<Status>);
          result->status_set = false;
        }
        result->status_set = true;
        result->status->set_value(st);
      }
      std::cout << "WARN: ExecuteLoop encountered error, status[" << st.ToString() << "]";
      return;
    }
    spec->id = (spec->id + 1) % spec->outputs.size();
    ExecuteLoopImpl(spec);
  });
}

void ExecuteLoop(const GraphDef& def, const std::vector<OutputSpec>& outputs) {
  ExecuteLoopSpec* spec = new ExecuteLoopSpec{.def = def, .outputs = outputs, .id = 0};
  ExecuteLoopImpl(spec);
}

Status ExecuteLoopWait() {
  Status ret;
  ExecuteLoopResult* result = ExecuteLoopResult::Instance();
  {
    std::future<Status> status;
    {
      std::unique_lock<std::mutex> lock(result->mu);
      if (result->status == nullptr) {
        result->status.reset(new std::promise<Status>);
        result->status_set = false;
      }
      status = result->status->get_future();
    }
    status.wait();
    ret = status.get();
  }
  {
    std::unique_lock<std::mutex> lock(result->mu);
    result->status.reset(nullptr);
  }
  return ret;
}

struct ExecutorContextWrapper {
  std::shared_ptr<ExecutorContext> internal;
  ExecutorContextWrapper(int size) : internal(std::make_shared<ExecutorContext>(size)) {}
};

void ExecutorPybind(pybind11::module& m) {
  pybind11::enum_<DataType>(m, "DataType")
    .value("int8", DataType::kInt8)
    .value("int16", DataType::kInt16)
    .value("int32", DataType::kInt32)
    .value("int64", DataType::kInt64)
    .value("float", DataType::kFloat)
    .value("double", DataType::kDouble)
    .value("bool", DataType::kBool);

  pybind11::class_<TensorShape>(m, "TensorShape")
    .def(pybind11::init<std::vector<size_t>>())
    .def("__len__", &TensorShape::Size)
    .def("__getitem__", &TensorShape::operator[]);

  pybind11::class_<DeviceDef>(m, "DeviceDef")
    .def(pybind11::init<>())
    .def_readwrite("device_name", &DeviceDef::device_name)
    .def_readwrite("attr", &DeviceDef::attr);

  pybind11::class_<AttrValue> attr_value(m, "AttrValue");
  attr_value
    .def(pybind11::init<>())
    .def_readwrite("s", &AttrValue::s)
    .def_readwrite("i", &AttrValue::i)
    .def_readwrite("f", &AttrValue::f)
    .def_readwrite("b", &AttrValue::b)
    .def_readwrite("shape", &AttrValue::shape)
    .def_readwrite("type", &AttrValue::type)
    .def_readwrite("type_list", &AttrValue::type_list)
    .def_readwrite("attr_type", &AttrValue::attr_type);

  pybind11::enum_<AttrValue::Type>(attr_value, "Type")
    .value("none", AttrValue::kNone)
    .value("string", AttrValue::kString)
    .value("int", AttrValue::kInt)
    .value("float", AttrValue::kFloat)
    .value("bool", AttrValue::kBool)
    .value("shape", AttrValue::kTensorShape)
    .value("type", AttrValue::kDataType)
    .value("type_list", AttrValue::kDataTypeList);

  pybind11::class_<NodeDef>(m, "NodeDef")
    .def(pybind11::init<>())
    .def(pybind11::init<const NodeDef&>())
    .def_readwrite("name", &NodeDef::name)
    .def_readwrite("op", &NodeDef::op)
    .def_readwrite("input", &NodeDef::input)
    .def_readwrite("output_type", &NodeDef::output_type)
    .def_readwrite("device", &NodeDef::device)
    .def_readwrite("attr", &NodeDef::attr);

  pybind11::class_<ExecutorContextWrapper>(m, "ExecutorContext")
    .def(pybind11::init<int>());

  pybind11::class_<RunOption>(m, "RunOption")
    .def(pybind11::init<bool>())
    .def(pybind11::init<>())
    .def("set_in_ctx",
      [](RunOption& a, const ExecutorContextWrapper& b) {
        a.in_ctx = b.internal.get();
      })
    .def("set_out_ctx",
      [](RunOption& a, const ExecutorContextWrapper& b) {
        a.out_ctx = b.internal.get();
      })
    .def_readwrite("perf", &RunOption::perf);

  pybind11::class_<GraphDef>(m, "GraphDef")
    .def(pybind11::init<>())
    .def_readwrite("node", &GraphDef::node)
    .def_readwrite("hash", &GraphDef::hash)
    .def("to_proto_string", &GraphDef::ToProtoString);

  pybind11::class_<OutputSpec>(m, "OutputSpec")
    .def(pybind11::init<>())
    .def_readwrite("output", &OutputSpec::output)
    .def_readwrite("output_device", &OutputSpec::output_device);

  pybind11::class_<RunStatistic>(m, "RunStatistic")
    .def(pybind11::init<>())
    .def_readwrite("perf_result", &RunStatistic::perf_result);    

  pybind11::class_<ExecuteResult>(m, "ExecuteResult")
    .def_readwrite("status", &ExecuteResult::status)
    .def_readwrite("outputs", &ExecuteResult::outputs)
    .def_readwrite("run_statistic", &ExecuteResult::run_statistic);

  pybind11::bind_vector<std::vector<NodeDef>>(
      m, "NodeDefVector");

  pybind11::bind_vector<std::vector<DataType>>(
      m, "DataTypeVector");

  pybind11::bind_vector<std::vector<OutputSpec>>(
      m, "OutputSpecVector");

  pybind11::bind_map<std::unordered_map<std::string, AttrValue>>(
      m, "StringAttrValueMap");

  m.def("execute", &Execute, "Execute the GraphDef");

  m.def("execute_loop", &ExecuteLoop, "Execute the GraphDef on loop");

  m.def("execute_loop_wait", &ExecuteLoopWait, "Wait execute_loop error");
}

}  // namespace python_lib
}  // namespace xdl

