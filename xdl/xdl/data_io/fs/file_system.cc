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

#include "xdl/data_io/fs/file_system.h"

#include <string.h>
#include <map>

#include "xdl/data_io/fs/file_system_hdfs.h"
#include "xdl/data_io/fs/file_system_local.h"
#include "xdl/data_io/fs/file_system_kafka.h"
#include "xdl/data_io/fs/zlib_ant.h"
#include "xdl/core/utils/logging.h"

namespace xdl {
namespace io {

/// The file system gfs map
std::map<int, FileSystem*> gfs;

std::string FileSystem::Read(const std::string &path) {
  std::unique_ptr<IOAnt> io(GetAnt(path.c_str()));
  size_t size = Size(path.c_str());
  size_t len = std::min(size, kDirectReadSize);
  std::string ret;
  ret.resize(len);
  XDL_CHECK(io->Read(&ret[0], len) > 0) << "read failed, path=" << path;
  ret.resize(len);
  return ret;
}

bool FileSystem::Write(const std::string &path, const std::string &content) {
  std::unique_ptr<IOAnt> io(GetAnt(path.c_str(), 'w'));
  size_t len = content.size();
  XDL_CHECK(len == io->Write(content.data(), len)) << "write failed, path=" << path;
  return true;
}

IOAnt *FileSystem::GetZAnt(const char *path, ZType ztype) {
  auto *ant = GetAnt(path);
  if (ztype == kZLib) {
    ant = new ZlibAnt(ant);
  }
  return ant;
}

/*!\brief Get the related filesystem */
FileSystem *GetFileSystem(FSType type, const char *ext) {
  if (gfs.find(type) != gfs.end()) {
    return gfs[type];
  }

  FileSystem* fs = nullptr;
  if (type == kHdfs) {
    fs = FileSystemHdfs::Get(ext);
  } else if (type == kLocal) {
    fs = FileSystemLocal::Get();
  } else if (type == kKafka) {
    fs = FileSystemKafka::Get(ext);
  } else {
    XDL_LOG(ERROR) << "Unkown file system, type=" << type;
  }
  XDL_CHECK(fs != nullptr) << "fs is nullptr";
  gfs[type] = fs;

  return fs;
}

}  // namespace io
}  // namespace xdl
