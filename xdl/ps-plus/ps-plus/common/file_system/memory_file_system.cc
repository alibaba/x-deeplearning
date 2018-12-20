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

#include "ps-plus/common/file_system.h"

#include <mutex>
#include <unordered_map>
#include <string>
#include <cstring>

namespace ps {

class MemoryFileSystem : public FileSystem {
 public:
  class MemoryReadStream : public ReadStream {
   public:
    MemoryReadStream(std::string* str) : str_(str), offset_(0) {}
    virtual ~MemoryReadStream() {Close();}
    virtual int64_t ReadSimple(void* buf, size_t size) override {
      if (offset_ + size > str_->size()) {
        size = str_->size() - offset_;
      }
      memcpy(buf, str_->data() + offset_, size);
      offset_ += size;
      return size;
    }
   protected:
    virtual void CloseInternal() override {
    }
    std::string* str_;
    size_t offset_;
  };

  class MemoryWriteStream : public WriteStream {
   public:
    MemoryWriteStream(std::string* str) : str_(str) {}
    virtual ~MemoryWriteStream() {Close();}
    virtual int64_t WriteSimple(const void* buf, size_t size) override {
      str_->append((const char*)buf, size);
      return size;
    }
    virtual void Flush() override {
    }
   protected:
    virtual void CloseInternal() {
    }
    std::string* str_;
  };

  virtual Status OpenReadStream(const std::string& name, ReadStream** result) override {
    std::unique_lock<std::mutex> lock(mu_);
    if (strs_.find(name) == strs_.end()) {
      return Status::NotFound("File Not Found");
    }
    *result = new MemoryReadStream(strs_[name].get());
    return Status::Ok();
  }

  virtual Status OpenWriteStream(const std::string& name, WriteStream** result, bool append = false) override {
    std::unique_lock<std::mutex> lock(mu_);
    if (strs_[name] == nullptr) {
      strs_[name].reset(new std::string);
    }
    *result = new MemoryWriteStream(strs_[name].get());
    return Status::Ok();
  }

  virtual Status Mkdir(const std::string& name) override {
    return Status::Ok();
  }

  virtual Status ListDirectory(const std::string& name, std::vector<std::string>* results) override {
    return Status::Ok();
  }

  virtual Status Remove(const std::string& name) override {
    std::unique_lock<std::mutex> lock(mu_);
    strs_.erase(name);
    return Status::Ok();
  }

  virtual Status Rename(const std::string& src_name, const std::string& dst_name) override {
    std::unique_lock<std::mutex> lock(mu_);
    strs_[dst_name] = std::move(strs_[src_name]);
    strs_.erase(src_name);
    return Status::Ok();
  }
  
  std::mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<std::string>> strs_;
};

PLUGIN_REGISTER(FileSystem, memory, MemoryFileSystem);

}

