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

#ifndef PS_SERVICE_SEASTAR_LIB_SERVER_FUNC_MANAGER_H_
#define PS_SERVICE_SEASTAR_LIB_SERVER_FUNC_MANAGER_H_

#include <functional>
#include <mutex>

namespace ps {
class Data;

namespace service {
namespace seastar {

class DoneClosure;
using ServerFunc = std::function<void(const std::vector<ps::Data*>& request_datas,
                                      std::vector<ps::Data*>* response_datas,
                                      DoneClosure* done)>;

class ServerFuncManager {
 public:
  static ServerFuncManager* GetInstance() {
    static ServerFuncManager sInstance;
    return &sInstance;
  }

  int RegisterServerFunc(size_t id, const ServerFunc& server_func) {
    bool exist = server_funcs_.insert({id, server_func}).second;
    if (exist) {
      return -1;
    }

    return 0;
  }

  int RegisterServerFunc(size_t id, const ServerFunc&& server_func) {
    bool exist = server_funcs_.emplace(id, std::move(server_func)).second;
    if (exist) {
      return -1;
    }

    return 0;
  }

  int GetServerFunc(size_t id, ServerFunc* server_func) {
    auto it = server_funcs_.find(id);
    if (it == server_funcs_.end()) {
      return -1;
    }

    *server_func = it->second;
    return 0;
  }

 private:
  ServerFuncManager() {}
  ~ServerFuncManager() {}    

 private:
  std::unordered_map<size_t, ServerFunc> server_funcs_;
};

} // namespace seastar
} // namespace service
} // namespace ps

#endif //PS_SERVICE_SEASTAR_LIB_SERVER_FUNC_MANAGER_H_
