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

#include "ps-plus/common/logging.h"
#include "ps-plus/common/option_parser.h"
#include "ps-plus/server/server_service.h"
#include "ps-plus/scheduler/scheduler_impl.h"
#include "ps-plus/scheduler/placementer.h"

#include <dlfcn.h>
#include <thread>

int ServerRun(int argc, char** argv) {
  ps::OptionParser optParser;
  optParser.addOption("-sp", "--scheduler_kv_path", "scheduler_kv_path", ps::OptionParser::OPT_STRING, true);
  optParser.addOption("-si", "--server_id", "server_id", ps::OptionParser::OPT_INT32, true);
  optParser.addOption("-smdense", "--streaming_model_dense", "streaming_model_dense", "");
  optParser.addOption("-smsparse", "--streaming_model_sparse", "streaming_model_sparse", "");
  optParser.addOption("-smhash", "--streaming_model_hash", "streaming_model_hash", "");
  optParser.addOption("-bc", "--bind_cores", "bind_cores", ps::OptionParser::OPT_STRING, true);
  if (!optParser.parseArgs(argc, argv)) {
    LOG(ERROR) << "Parse Server Args Error";
    return -1;
  }

  std::string scheduler_kv_path;
  int server_id;
  std::string streaming_model_dense;
  std::string streaming_model_sparse;
  std::string streaming_model_hash;
  std::string bind_cores;
  
  optParser.getOptionValue("scheduler_kv_path", scheduler_kv_path);
  optParser.getOptionValue("server_id", server_id);
  optParser.getOptionValue("streaming_model_dense", streaming_model_dense);
  optParser.getOptionValue("streaming_model_sparse", streaming_model_sparse);
  optParser.getOptionValue("streaming_model_hash", streaming_model_hash);
  optParser.getOptionValue("bind_cores", bind_cores);

  ps::server::ServerService service(scheduler_kv_path, server_id,
                                    streaming_model_dense, streaming_model_sparse, streaming_model_hash,
                                    bind_cores == "True" ? true : false);
  ps::Status st = service.Init();
  if (!st.IsOk()) {
    LOG(ERROR) << "ERROR ON Server Init: " << st.ToString();
    return -1;
  }
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(3600));
  }
  return 0;
}

int SchedulerRun(int argc, char** argv) {
  ps::OptionParser optParser;
  optParser.addOption("-sp", "--scheduler_kv_path", "scheduler_kv_path", ps::OptionParser::OPT_STRING, true);
  optParser.addOption("-sn", "--server_num", "server_num", ps::OptionParser::OPT_STRING, true);
  optParser.addOption("-snet", "--server_network_limit", "server_network_limit", ps::OptionParser::OPT_INT32, true); //MB/s
  optParser.addOption("-smem", "--server_memory_limit", "server_memory_limit", ps::OptionParser::OPT_INT32, true); //MB
  optParser.addOption("-sqps", "--server_query_limit", "server_query_limit", ps::OptionParser::OPT_INT32, true);
  optParser.addOption("-cp", "--checkpoint_path", "checkpoint_path", "none://");
  optParser.addOption("-smdense", "--streaming_model_dense", "streaming_model_dense", "");
  optParser.addOption("-smsparse", "--streaming_model_sparse", "streaming_model_sparse", "");
  optParser.addOption("-smhash", "--streaming_model_hash", "streaming_model_hash", "");
  optParser.addOption("-bc", "--bind_cores", "bind_cores", ps::OptionParser::OPT_STRING, true);

  if (!optParser.parseArgs(argc, argv)) {
    LOG(ERROR) << "argument error";
    return -1;
  }

  std::string scheduler_kv_path;
  std::string checkpoint_path;
  std::string server_num;
  int server_network_limit;
  int server_memory_limit;
  int server_query_limit;
  std::string streaming_model_dense;
  std::string streaming_model_sparse;
  std::string streaming_model_hash;
  std::string bind_cores;

  optParser.getOptionValue("scheduler_kv_path", scheduler_kv_path);
  optParser.getOptionValue("checkpoint_path", checkpoint_path);
  optParser.getOptionValue("server_num", server_num);
  optParser.getOptionValue("server_network_limit", server_network_limit);
  optParser.getOptionValue("server_memory_limit", server_memory_limit);
  optParser.getOptionValue("server_query_limit", server_query_limit);
  optParser.getOptionValue("streaming_model_dense", streaming_model_dense);
  optParser.getOptionValue("streaming_model_sparse", streaming_model_sparse);
  optParser.getOptionValue("streaming_model_hash", streaming_model_hash);
  optParser.getOptionValue("bind_cores", bind_cores);

  ps::scheduler::Placementer::Arg placement_arg {
    .net = (size_t)server_network_limit * (1 << 20),
    .mem = (size_t)server_memory_limit * (1 << 20),
    .query = (size_t)server_query_limit
  };

  ps::scheduler::SchedulerImpl service(
      server_num, scheduler_kv_path, checkpoint_path, placement_arg,
      streaming_model_dense, streaming_model_sparse, streaming_model_hash, bind_cores == "True" ? true : false);
  ps::Status st = service.Start();
  if (!st.IsOk()) {
    LOG(ERROR) << "ERROR ON Server Init: " << st.ToString();
    return -1;
  }
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(3600));
  }
  return 0;
}

int LoadPlugin(const std::string& name) {
  void* handle = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    std::cout << "dlopen error: " << dlerror() << std::endl;
    return -1;
  }
  return 0;
}

int LoadPlugins(int argc, char** argv) {
  std::vector<std::string> plugins;
  for (int i = 0; i < argc - 1; i++) {
    std::string arg = argv[i];
    if (arg == "--plugin") {
      plugins.push_back(argv[i + 1]);
    }
  }
  for (auto& plugin : plugins) {
    int ret = LoadPlugin(plugin);
    if (ret != 0) {
      return ret;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  int ret = LoadPlugins(argc, argv);
  if (ret != 0) {
    return ret;
  }

  ps::OptionParser optParser;
  optParser.addOption("-r", "--role", "role", ps::OptionParser::OPT_STRING, true);
  std::string role;
  if (!optParser.parseArgs(argc, argv)) {
    LOG(ERROR) << "Must Specify role";
    return -1;
  }

  optParser.getOptionValue("role", role);
  if (role == "scheduler") {
    return SchedulerRun(argc, argv);
  } else if (role == "server") {
    return ServerRun(argc, argv);
  } else {
    LOG(ERROR) << "Role must be scheduler or server";
    return -1;
  }
}
