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

#include <assert.h>
#include "xdl/core/utils/logging.h"
#include "ps-plus/common/option_parser.h"
#include "ps-plus/common/initializer.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/client/client.h"
#include "ps-plus/client/client_wrapper_impl.h"

using namespace ps;

#define XX 40
#define YY 80

void callBack(const Status& status) {
  if (!status.IsOk()) {
    std::cerr << status.ToString() << std::endl;
  }
}

Status registVariable(client::Client& client, const std::string& variable_name, size_t i = 200) {
  VariableInfo info;
  info.type = VariableInfo::kIndex;
  info.name = variable_name;
  info.shape = {XX * (int64_t)i + XX, YY * (int64_t)i + YY};
  info.datatype = ps::types::DataType::kInt64;
  return client.RegisterVariable(variable_name, info);
}

void densePushVariable(client::Client& client, const std::string& variable_name, size_t i = 200) {
  Tensor grad(DataType::kInt64, TensorShape({XX * i + XX, YY * i + YY}), new initializer::ConstantInitializer(i));
  std::vector<Data*> datas = client.Args(grad);  
  std::promise<bool> p;
  client.DensePush(variable_name, "AssignAddUpdater", datas, [&](const Status& st) {
    if (!st.IsOk()) {
      std::cout << st.ToString() << std::endl;
    }
    p.set_value(true);
  });
  
  p.get_future().wait();
}

void initVariable(client::Client& client, const std::string& variable_name) {
  Initializer* initializer = new initializer::ConstantInitializer(2);
  std::promise<bool> p;
  client.IndexInitializer(variable_name, initializer, [&](const Status& st) {
    if (!st.IsOk()) {
      std::cout << st.ToString() << std::endl;
    } 
    p.set_value(true);
  });

  p.get_future().wait();
}

void densePullVariable(client::Client& client, const std::string& variable_name) {
  Tensor result;
  std::promise<bool> p;
  client.DensePull(variable_name, &result, [&](const Status& st) {
    if (!st.IsOk()) {
      std::cout << st.ToString() << std::endl;
    } else {
      int64_t* tensor_data = result.Raw<int64_t>();
      std::cout << "pull result:" << result.Shape().NumElements() << " * " << tensor_data[0] << std::endl;
      int xx = 0;
      for (size_t i = 0; i < result.Shape().NumElements(); i++) {
        if (tensor_data[i] != tensor_data[0]) {
          std::cout << "ERROR At " << i << " with " << tensor_data[i] << " and " << tensor_data[i + 1] << " and " << tensor_data[0] << std::endl;
          if (xx++ > 1000) {
          break;
          }
        }
      }
    }
    p.set_value(true);
  });
  p.get_future().wait();
}

int main(int argc, char** argv) {
  OptionParser optParser;
  optParser.addOption("-v", "--variable_name", "variable_name", OptionParser::OPT_STRING, true);
  optParser.addOption("-sn", "--server_num", "server_num", OptionParser::OPT_INT32, true);
  optParser.addOption("-sp", "--scheduler_kv_path", "scheduler_kv_path", OptionParser::OPT_STRING, true);
  optParser.addOption("-a", "--action", "action", OptionParser::OPT_STRING, true);  
  if (!optParser.parseArgs(argc, argv)) {
    LOG(ERROR) << "Parse Server Args Error";
    return -1;
  }  

  std::string variable_name;
  int32_t server_num;
  std::string scheduler_kv_path;
  std::string action;
  optParser.getOptionValue("variable_name", variable_name);
  optParser.getOptionValue("server_num", server_num);
  optParser.getOptionValue("scheduler_kv_path", scheduler_kv_path);
  optParser.getOptionValue("action", action);
  
  client::ClientArgs args;
  args.scheduler_addr = scheduler_kv_path;
  args.client_wrapper_creator = [](){return new client::ClientWrapperImpl();};
  client::RawClient* raw_client = new client::RawClient(args);
  client::Client client(raw_client);
  client.Init();

  if (action == "test") {
    registVariable(client, variable_name);
    initVariable(client, variable_name);
    densePushVariable(client, variable_name);
    densePullVariable(client, variable_name);
  } else if (action == "push") {
    densePushVariable(client, variable_name);
  } else if (action == "pull") {
    densePullVariable(client, variable_name);
  } else {
    for (int i = 0; i < 10; i++) {
      registVariable(client, variable_name + "_" + std::to_string(i), (size_t)i);
      initVariable(client, variable_name + "_" + std::to_string(i));
      densePushVariable(client, variable_name + "_" + std::to_string(i), (size_t)i);
    }
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
      threads.emplace_back([&, i](){
        while (true) {
          densePullVariable(client, variable_name + "_" + std::to_string(i));
        }
      });
    }
    for (int i = 0; i < 10; i++) {
      threads[i].join();
    }
  }
}
