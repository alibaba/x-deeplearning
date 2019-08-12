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

#include "xdl/data_io/fs/file_system_local.h"

#include <dirent.h>
#include <memory>
#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>

#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

/// Local IO ant
class IOAntLocal : public IOAnt {
 public:
  IOAntLocal(FILE *fd) : fd_(fd) { XDL_CHECK(fd != nullptr); }
  ~IOAntLocal() { if (fd_) fclose(fd_); }

  /*!\brief read data */
  virtual ssize_t Read(char *data, size_t len) override {
    return fread(data, 1, len, fd_);
  }
  /*!\brief seek to offset */
  virtual ssize_t Write(const char *data, size_t len) override {
    return fwrite(data, 1, len, fd_);
  }

  /*!\brief seek to offset */
  virtual off_t Seek(off_t offset) override {
    return fseek(fd_, offset, SEEK_SET);
  }

 protected:
  /*! read only fd */
  FILE *fd_;
};

IOAnt *FileSystemLocal::GetAnt(const char *path, char mode) {
  FILE *fd = mode == 'r' ? fopen(path, "r") : fopen(path, "w");
  XDL_CHECK(fd != nullptr) << "Open " << path << " failed";
  return new IOAntLocal(fd);
}

bool FileSystemLocal::IsReg(const char *path) {
  struct stat stat;
  if (lstat(path, &stat) == -1) {
    return false;
  }

  return S_ISREG(stat.st_mode);
}

bool FileSystemLocal::IsDir(const char *path) {
  struct stat stat;
  if (lstat(path, &stat) == -1) {
    return false;
  }

  return S_ISDIR(stat.st_mode);
}

std::vector<std::string> FileSystemLocal::Dir(const char *path) {
  std::vector<std::string> paths;
  DIR *dp;
  struct dirent *dirp;
  
  dp = opendir(path);
  XDL_CHECK(dp != nullptr) << "can't open dir "<< path;

  while ((dirp = readdir(dp)) != nullptr) {
    paths.push_back(dirp->d_name);
  }
  
  closedir(dp);

  return paths;
}

void *FileSystemLocal::Open(const char *path, const char *mode) {
  FILE *fd = fopen(path, "r");
  XDL_CHECK(fd != nullptr) << "can't open " << path;
  return fd;
}

size_t FileSystemLocal::Size(const char *path) {
  struct stat stat;
  XDL_CHECK(lstat(path, &stat) != -1) << "can't stat " << path;
  return stat.st_size;
}

/// LocalFileSystem Implementation
FileSystemLocal::~FileSystemLocal() { }

FileSystem *FileSystemLocal::Get() {
  static std::unique_ptr<FileSystemLocal> inst(new FileSystemLocal());
  return inst.get();
}
  
}  // namespace io
}  // namespace xdl
