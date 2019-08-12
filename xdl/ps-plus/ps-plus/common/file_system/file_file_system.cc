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

#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>  

namespace ps {
class FileFileSystem : public FileSystem {
 public:
  class FileReadStream : public ReadStream {
   public:
    FileReadStream(FILE* file) : file_(file) {}
    virtual ~FileReadStream() {Close();}
    virtual int64_t ReadSimple(void* buf, size_t size) override {
      return fread(buf, 1, size, file_);
    }
   protected:
    virtual void CloseInternal() override {
      fclose(file_);
    }
   private:
    FILE* file_;
  };

  class FileWriteStream : public WriteStream {
   public:
    FileWriteStream(FILE* file) : file_(file) {}
    virtual ~FileWriteStream() {Close();}
    virtual int64_t WriteSimple(const void* buf, size_t size) override {
      return fwrite(buf, 1, size, file_);
    }
    virtual void Flush() override {
      fflush(file_);
    }
   protected:
    virtual void CloseInternal() {
      fclose(file_);
    }
   private:
    FILE* file_;
  };

  virtual Status OpenReadStream(const std::string& name, ReadStream** result) override {
    std::string real_name = name.substr(0, 7) == "file://" ? name.substr(7) : name;
    FILE *file = fopen(real_name.c_str(), "rb");
    if (file == nullptr) {
      return Status::Unknown("File open Error " + name);
    }
    *result = new FileReadStream(file);
    return Status::Ok();
  }

  virtual Status OpenWriteStream(const std::string& name, WriteStream** result, bool append = false) override {
    std::string real_name = name.substr(0, 7) == "file://" ? name.substr(7) : name;
    auto pos = real_name.find_last_of('/');
    if (pos != std::string::npos) {
      PS_CHECK_STATUS(Mkdir(real_name.substr(0, pos)));
    }
    FILE *file = fopen(real_name.c_str(), append ? "ab" : "wb");
    if (file == nullptr) {
      return Status::Unknown("File open Error " + name);
    }
    *result = new FileWriteStream(file);
    return Status::Ok();
  }

  virtual Status Mkdir(const std::string& name) override {
    std::string real_dir = name.substr(0, 7) == "file://" ? name.substr(7) : name;
    if (real_dir.empty()) {
      return Status::NotFound("Mkdir must not be root");
    }
    real_dir = real_dir.back() == '/' ? real_dir : real_dir + '/';
    for (size_t i = 1; i < real_dir.size(); i++) {
      if (real_dir[i] == '/') {
        std::string current_dir = real_dir.substr(0, i);
        if(access(current_dir.c_str(), F_OK) != 0) {
          if(mkdir(current_dir.c_str(), 0755) != 0) {
            if(access(current_dir.c_str(), F_OK) != 0) {
              return Status::Unknown("Mkdir Error " + current_dir);
            }
          }
        }
      }
    }
    return Status::Ok();
  }

  virtual Status ListDirectory(const std::string& name, std::vector<std::string>* results) override {
    std::string real_dir = name.substr(0, 7) == "file://" ? name.substr(7) : name;
    DIR *pDir;  
    struct dirent *ent;
    char absolutepath[512] = {0};
    pDir = opendir(real_dir.c_str());
    while ((ent = readdir(pDir)) != NULL) {
      if (ent->d_type & DT_DIR) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
          continue;
        } else {
          sprintf(absolutepath, "%s/%s", real_dir.c_str(), ent->d_name);
          results->push_back(absolutepath);
        }
      }
    }
    return Status::Ok();
  }

  virtual Status Remove(const std::string& name) override {
    std::string real_name = name.substr(0, 7) == "file://" ? name.substr(7) : name;
    if (remove(real_name.c_str()) != 0) {
      return Status::Unknown("Remove Error");
    }
    return Status::Ok();
  }

  virtual Status Rename(const std::string& src_name, const std::string& dst_name) override {
    std::string real_src = src_name.substr(0, 7) == "file://" ? src_name.substr(7) : src_name;
    std::string real_dst = dst_name.substr(0, 7) == "file://" ? dst_name.substr(7) : dst_name;
    if (rename(real_src.c_str(), real_dst.c_str()) != 0) {
      return Status::Unknown("Rename Error");
    }
    return Status::Ok();
  }
};

PLUGIN_REGISTER(FileSystem, file, FileFileSystem);

}

