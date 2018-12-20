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

namespace ps {

class NoneFileSystem : public FileSystem {
 public:
  class NoneReadStream : public ReadStream {
   public:
    virtual ~NoneReadStream() {Close();}
    virtual int64_t ReadSimple(void* buf, size_t size) override {
      return 0;
    }
   protected:
    virtual void CloseInternal() override {
    }
  };

  class NoneWriteStream : public WriteStream {
   public:
    virtual ~NoneWriteStream() {Close();}
    virtual int64_t WriteSimple(const void* buf, size_t size) override {
      return 0;
    }
    virtual void Flush() override {
    }
   protected:
    virtual void CloseInternal() {
    }
  };

  virtual Status OpenReadStream(const std::string& name, ReadStream** result) override {
    return Status::NotFound("not found");
  }

  virtual Status OpenWriteStream(const std::string& name, WriteStream** result, bool append = false) override {
    *result = new NoneWriteStream;
    return Status::Ok();
  }

  virtual Status Mkdir(const std::string& name) override {
    return Status::Ok();
  }

  virtual Status ListDirectory(const std::string& name, std::vector<std::string>* results) override {
    return Status::Ok();
  }

  virtual Status Remove(const std::string& name) override {
    return Status::Ok();
  }

  virtual Status Rename(const std::string& src_name, const std::string& dst_name) override {
    return Status::Ok();
  }
};

PLUGIN_REGISTER(FileSystem, none, NoneFileSystem);

}

