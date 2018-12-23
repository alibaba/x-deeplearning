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

#include "seastar_server_lib.h"

#include "cpu_pool.h"
#include "server_func_manager.h"

namespace ps {
namespace service {
namespace seastar {

SeastarServerLib::SeastarServerLib(int port,
                                   int core_num)
  : server_context_(1, 1, core_num, 1)
  , port_(port)
  , core_num_(core_num) {
}

bool SeastarServerLib::Start() {
  if (!CPUPool::GetInstance()->Allocate(core_num_, &core_ids_)) {
    std::cerr << "allocate core[" << core_num_ << "] failed!" << std::endl;
    return false;
  }
  
  int argc;
  char** argv;
  ToCmdOptions(&argc, &argv);
  server_thread_.reset(new std::thread([this, argc, argv] {
    server_context_(argc, argv);
  }));
  return true;
}

void SeastarServerLib::ToCmdOptions(int* argc, char*** argv) {
  std::string is_poll_mode = ps::NetUtils::GetEnv("POLL_MODE");
  if (is_poll_mode == "1") {
    *argc = 9;
    *argv = new char*[10];
  } else {
    *argc = 8;
    *argv = new char*[9];
  }

  (*argv)[0] = new char[1000];
  (*argv)[1] = new char[1000];
  (*argv)[2] = new char[1000];
  (*argv)[3] = new char[1000];
  (*argv)[4] = new char[1000];
  (*argv)[5] = new char[1000];
  (*argv)[6] = new char[1000];
  (*argv)[7] = new char[1000];
  (*argv)[8] = nullptr;
  
  strcpy((*argv)[0], "fake_path_for_seastar");
  snprintf((*argv)[1], 1000, "--smp=%d", core_num_);
  snprintf((*argv)[2], 1000, "--cpuset=%s", core_ids_.c_str());
  snprintf((*argv)[3], 1000, "--port=%d", port_);
  strcpy((*argv)[4], "--tcp_nodelay_on=1");
  strcpy((*argv)[5], "--tcp_keep_alive_idle=300");
  strcpy((*argv)[6], "--tcp_keep_alive_cnt=6");
  strcpy((*argv)[7], "--tcp_keep_alive_interval=10");
  if (is_poll_mode == "1") {
    (*argv)[8] = new char[1000];
    strcpy((*argv)[8], "--poll-mode");
    (*argv)[9] = nullptr;
  }
}

void SeastarServerLib::Stop() {
  server_thread_->join();
}

bool SeastarServerLib::RegisterServerFunc(size_t id, 
                                          const ServerFunc& server_func) {
  return ServerFuncManager::GetInstance()->RegisterServerFunc(
      id, server_func) == 0;
}

} // namespace seastar
} // namespace service
} // namespace ps
