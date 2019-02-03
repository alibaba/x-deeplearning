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

#include "xdl/core/inference/serving.h"

#include "xdl/core/utils/logging.h"
#include "xdl/core/utils/file_utils.h"

namespace xdl {

Serving::Serving(const std::string& ckpt_dir)
  : ckpt_dir_(ckpt_dir)
  , client_(nullptr) {
  executor_.reset(new Executor(ThreadPool::Global()));
}

Status Serving::Init(const std::string& graph_path,
                     const std::string& ckpt_version) {
  XDL_CHECK_COND(
      ConnectToClient("localhost", ckpt_dir_),
      Status::Internal("connect to client failed"));
  XDL_CHECK_COND(
      GetClient(&client_).IsOk(),
      Status::Internal("get client failed"));

  XDL_CHECK_STATUS(LoadGraph(graph_path));
  std::promise<bool> p;
  auto cb = [&p](const ps::Status& st) {
    bool ret = st.IsOk() ? true : false;
    if (!ret) {
      XDL_LOG(ERROR) << st.ToString();
    }
    
    p.set_value(ret);
  };

  client_->Restore(ckpt_version, cb);
  bool ret = p.get_future().get();
  if (!ret) {
    return Status::Internal("local_client restore failed");
  }

  return Status::Ok();
}

Status Serving::LoadGraph(
    const std::string& graph_path,
    bool text_format) {
  std::string graph_str = FileUtils::ReadLocalBinaryFile(graph_path);
  XDL_CHECK_COND(
      !graph_str.empty(),
      Status::ArgumentError("failed to read graph:" + graph_path));
  if (text_format) {
    XDL_CHECK_COND(
        graph_.FromTextString(graph_str),
        Status::ArgumentError("cant't parse graph"));
  } else {
    XDL_CHECK_COND(
        graph_.FromProtoString(graph_str),
        Status::ArgumentError("cant't parse graph"));
  }

  XDL_CHECK_STATUS(ParseInferenceTag());
  return Status::Ok();
}

Status Serving::ParseInferenceTag() {
  for (size_t i = 0; i < graph_.tag.inputs.size(); ++i) {
    const InputDef& input= graph_.tag.inputs[i];
    input_tag_.insert(make_pair(input.input_name, input.op_name));
  }

  for (size_t i = 0; i < graph_.tag.outputs.size(); ++i) {
    output_spec_.output.push_back(graph_.tag.outputs[i].op_name);
  }

  output_spec_.output_device.device_name = "CPU";
  return Status::Ok();
}

Status Serving::Predict(
    const Executor::Feeds& feeds, 
    std::vector<Tensor>* results) {
  XDL_CHECK_COND(
      !input_tag_.empty() && !output_spec_.output.empty(),
      Status::ArgumentError("no TagDef found in graph def"));

  Executor::Feeds real_feeds;
  real_feeds.reserve(feeds.size());
  for (auto& feed: feeds) {
    auto it = input_tag_.find(feed.first);
    XDL_CHECK_COND(
        it != input_tag_.end(),
        Status::ArgumentError("input tag not found:" + feed.first));
    real_feeds.push_back({it->second, feed.second});
  }

  return Predict(real_feeds, output_spec_, results);
}

Status Serving::Predict(
    const Executor::Feeds& feeds, 
    const std::vector<std::string>& output_op_names,
    std::vector<Tensor>* results) {
  OutputSpec output_spec;
  output_spec.output = output_op_names;
  output_spec.output_device.device_name = "CPU";
  return Predict(feeds, output_spec, results);
}

Status Serving::Predict(
    const Executor::Feeds& feeds,
    const OutputSpec output_spec,
    std::vector<Tensor>* results) {
  RunOption run_option;
  std::promise<bool> p;
  auto cb = [&p, results](
      Status st, 
      const std::vector<Tensor>& outputs, 
      const SimpleExecutor::ExtraInfo& exinfo) {
    bool ret = st.IsOk();
    if (ret) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        results->emplace_back(outputs[i]);
      }
    } else {
      printf("%s\n", st.ToString().c_str());
    }

    p.set_value(ret);
  };

  executor_->Run(graph_, feeds, output_spec, run_option, cb);
  if (p.get_future().get()) {
    return Status::Ok();
  } else {
    return Status::Internal("executor run failed");
  }
}

}  // namespace xdl

