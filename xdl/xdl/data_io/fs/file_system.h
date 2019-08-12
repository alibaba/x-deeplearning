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

#ifndef XDL_IO_FS_FILE_SYSTEM_H_
#define XDL_IO_FS_FILE_SYSTEM_H_

#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include "xdl/data_io/constant.h"

namespace xdl {
namespace io {

static const size_t kDirectReadSize = 256*1024;

/*!\brier the io ant */
class IOAnt {
 public:
  virtual ~IOAnt() = default;
  /*!\brief read data */
  virtual ssize_t Read(char* data, size_t len) = 0;
  virtual ssize_t Write(const char* data, size_t len) = 0;
  /*!\brief seek to offset */
  virtual off_t Seek(off_t offset) = 0;
  virtual off_t SeekRange(off_t begin, off_t end) {}

  /*! set ref */
  void set_ref(bool ref) { ref_ = ref; }
  /*! get ref */
  bool ref() const { return ref_; }

 protected:
  bool ref_ = false;
};


/*! \brief file system system interface */
class FileSystem {
 public:
  /*! \brief virtual destructor */
  virtual ~FileSystem() { }
  virtual IOAnt *GetAnt(const char *path, char mode='r') = 0;

  /// only support zlib with read
  IOAnt *GetZAnt(const char *path, ZType ztype);

  virtual bool IsDir(const char *path) = 0;
  virtual bool IsReg(const char *path) = 0;
  virtual std::vector<std::string> Dir(const char *path) = 0;
  virtual void *Open(const char *path, const char *mode) = 0;
  virtual size_t Size(const char *path) = 0;
  virtual std::string Read(const std::string &path);
  virtual bool Write(const std::string &path, const std::string &content);

 protected:
  std::mutex mutex_;
};

/*!\brief Get the related filesystem */
FileSystem *GetFileSystem(FSType type, const char *ext=nullptr);

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_FS_FILE_SYSTEM_H_