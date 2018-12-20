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

#include "ps-plus/common/reliable_kv.h"
#include "ps-plus/common/zk_wrapper.h"
#include <thread>

namespace ps {

class ZfsKV : public ReliableKV {
 public:
  virtual Status Read(const std::string& addr, std::string* val, int retry) override {
    std::string zkaddr, zknode;
    PS_CHECK_STATUS(SplitZkAddr(addr, &zkaddr, &zknode));
    ZkWrapper* zk = GetZk(zkaddr);
    if (zk == nullptr) {
      return Status::NetworkError("Cannot connect to zk");
    }
    while (true) {
      if (zk->GetData(zknode, *val)) {
        return Status::Ok();
      } else {
        if (--retry == 0) {
          return Status::NetworkError("Cannot Get zknode: " + addr);
        }
      }
    }
    // NonTerminate Loop
  }
  virtual Status Write(const std::string& addr, const std::string& val, int retry) override {
    std::string zkaddr, zknode;
    PS_CHECK_STATUS(SplitZkAddr(addr, &zkaddr, &zknode));
    ZkWrapper* zk = GetZk(zkaddr);
    if (zk == nullptr) {
      return Status::NetworkError("Cannot connect to zk");
    }
    while (true) {
      if (zk->Touch(zknode, val, true)) {
        return Status::Ok();
      } else {
        if (--retry == 0) {
          return Status::NetworkError("Cannot Set zknode: " + addr);
        }
      }
    }
    // NonTerminate Loop
  }
 private:

  Status SplitZkAddr(const std::string& addr, std::string* zkaddr, std::string* zknode) {
    if (addr.substr(0, 6) != "zfs://") {
      return Status::ArgumentError("Not a zk addr");
    }
    std::string simple_addr = addr.substr(6);
    size_t pos = simple_addr.find('/');
    if (pos == std::string::npos) {
      *zkaddr = simple_addr;
      *zknode = "/";
    } else {
      *zkaddr = simple_addr.substr(0, pos);
      *zknode = simple_addr.substr(pos);
    }
    return Status::Ok();
  }

  ZkWrapper* GetZk(const std::string& zkaddr) {
    std::unique_lock<std::mutex> lock(zks_mu_);
    auto iter = zks_.find(zkaddr);
    if (iter != zks_.end()) {
      return iter->second.get();
    }
    std::unique_ptr<ZkWrapper> zk(new ZkWrapper(zkaddr, 60));
    int retry = 3;
    while (true) {
      if (zk->Open()) {
        break;
      } else {
        if (--retry == 0) {
          return nullptr;
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));
      }
    }
    ZkWrapper* result = zk.release();
    zks_[zkaddr].reset(result);
    return result;
  }

  std::mutex zks_mu_;
  std::unordered_map<std::string, std::unique_ptr<ZkWrapper>> zks_;
};

PLUGIN_REGISTER(ReliableKV, zfs, ZfsKV);

}

