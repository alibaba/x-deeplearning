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
#include "ps-plus/plugins/hdfs/libhdfs.h"
#include <mutex>

namespace ps {
namespace hdfs {

class HdfsFileSystem : public FileSystem {
 public:
  class HdfsReadStream : public ReadStream {
   public:
    HdfsReadStream(LibHDFS* hdfs, hdfsFS fs, hdfsFile file) : hdfs_(hdfs), fs_(fs), file_(file)  {}
    virtual ~HdfsReadStream() {Close();}
    virtual int64_t ReadSimple(void* buf, size_t size) override {
      return hdfs_->hdfsRead(fs_, file_, buf, size);
    }
   protected:
    virtual void CloseInternal() override {
      hdfs_->hdfsCloseFile(fs_, file_);
    }
   private:
    LibHDFS* hdfs_;
    hdfsFS fs_;
    hdfsFile file_;
  };

  class HdfsWriteStream : public WriteStream {
   public:
    HdfsWriteStream(LibHDFS* hdfs, hdfsFS fs, hdfsFile file) : hdfs_(hdfs), fs_(fs), file_(file)  {}
    virtual ~HdfsWriteStream() {Close();}
    virtual int64_t WriteSimple(const void* buf, size_t size) override {
      return hdfs_->hdfsWrite(fs_, file_, buf, size);
    }
    virtual void Flush() override {
      hdfs_->hdfsHSync(fs_, file_);
    }
   protected:
    virtual void CloseInternal() {
      hdfs_->hdfsCloseFile(fs_, file_);
    }
   private:
    LibHDFS* hdfs_;
    hdfsFS fs_;
    hdfsFile file_;
  };

  HdfsFileSystem() : hdfs_(nullptr) {
    st_ = LibHDFS::Load(&hdfs_);
  }

  virtual Status OpenReadStream(const std::string& name, ReadStream** result) override {
    hdfsFS fs = nullptr;
    PS_CHECK_STATUS(GetFS(name, &fs));
    hdfsFile ret = hdfs_->hdfsOpenFile(fs, name.c_str(), O_RDONLY, 0, 0, 0);
    if (ret == nullptr) {
      return Status::Unknown("Open hdfs file for read Error: " + name);
    }
    *result = new HdfsReadStream(hdfs_, fs, ret);
    return Status::Ok();
  }

  virtual Status OpenWriteStream(const std::string& name, WriteStream** result, bool append = false) override {
    hdfsFS fs = nullptr;
    PS_CHECK_STATUS(GetFS(name, &fs));
    hdfsFile ret = nullptr;
    if (append) {
      ret = hdfs_->hdfsOpenFile(fs, name.c_str(), O_WRONLY|O_APPEND, 0, 0, 0);
    }
    if (ret == nullptr) {
      ret = hdfs_->hdfsOpenFile(fs, name.c_str(), O_WRONLY, 0, 0, 0);
    }
    if (ret == nullptr) {
      return Status::Unknown("Open hdfs file for write Error: " + name);
    }
    *result = new HdfsWriteStream(hdfs_, fs, ret);
    return Status::Ok();
  }

  virtual Status Mkdir(const std::string& name) override {
    hdfsFS fs = nullptr;
    PS_CHECK_STATUS(GetFS(name, &fs));
    int ret = hdfs_->hdfsCreateDirectory(fs, name.c_str());
    if (ret != 0) {
      return Status::Unknown("Create Directory Error " + name);
    }
    return Status::Ok();
  }

  virtual Status ListDirectory(const std::string& name, std::vector<std::string>* results) override {
    hdfsFS fs = nullptr;
    PS_CHECK_STATUS(GetFS(name, &fs));
    int num_entries;
    hdfsFileInfo* file_info = hdfs_->hdfsListDirectory(fs, name.c_str(), &num_entries);
    if (file_info == nullptr) {
      return Status::Unknown("List Directory Error " + name);
    }
    for (int i = 0; i < num_entries; ++i) {
      results->push_back(file_info[i].mName);
    }
    hdfs_->hdfsFreeFileInfo(file_info, num_entries);
    return Status::Ok();
  }

  virtual Status Remove(const std::string& name) override {
    hdfsFS fs = nullptr;
    PS_CHECK_STATUS(GetFS(name, &fs));
    hdfs_->hdfsDelete(fs, name.c_str(), false);
    return Status::Ok();
  }

  virtual Status Rename(const std::string& src_name, const std::string& dst_name) override {
    hdfsFS fs = nullptr;
    PS_CHECK_STATUS(GetFS(src_name, &fs));
    hdfs_->hdfsRename(fs, src_name.c_str(), dst_name.c_str());
    return Status::Ok();
  }
 private:
  Status GetFS(const std::string& name, hdfsFS* fs) {
    if (!st_.IsOk()) {
      return st_;
    }
    size_t pos = name.find("://");
    if (pos == std::string::npos) {
      return Status::ArgumentError("Hdfs Filename error " + name);
    }
    pos = name.find('/', pos + 3);
    if (pos == std::string::npos) {
      return Status::ArgumentError("Hdfs Filename error " + name);
    }
    std::string addr = name.substr(0, pos + 1);
    {
      std::unique_lock<std::mutex> lock(mu_);
      auto iter = hdfs_collenction_.find(addr);
      if (iter != hdfs_collenction_.end()) {
        *fs = iter->second;
        return Status::Ok();
      }
      hdfsBuilder* builder = hdfs_->hdfsNewBuilder();
      hdfs_->hdfsBuilderSetNameNode(builder, addr.c_str());
      hdfsFS ret = hdfs_->hdfsBuilderConnect(builder);
      if (ret == nullptr) {
        return Status::Unknown("Create Hdfs Error " + name);
      }
      *fs = hdfs_collenction_[addr] = ret;
      return Status::Ok();
    }
  }

  std::mutex mu_;
  std::unordered_map<std::string, hdfsFS> hdfs_collenction_;
  LibHDFS* hdfs_;
  Status st_;
};

PLUGIN_REGISTER(FileSystem, hdfs, HdfsFileSystem);

}
}

