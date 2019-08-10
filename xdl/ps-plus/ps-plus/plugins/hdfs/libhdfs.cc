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

#include <dlfcn.h>
#include <cstdlib>
#include <string>
#include <iostream>

#include "ps-plus/plugins/hdfs/libhdfs.h"

namespace ps {
namespace hdfs {

template<typename R, typename... Ts>
Status BindFunc(void* dl, const char* name, std::function<R (Ts...)>* const f) {
  void* sym = dlsym(dl, name);
  if (sym == nullptr) {
    return Status::NotFound(std::string("HDFS Symbol [") + name + "] Not Found");
  }
  *f = reinterpret_cast<R (*)(Ts...)>(sym);
  return Status::Ok();
}

Status LibHDFS::Load(LibHDFS** const result) {
  LibHDFS* lib = new LibHDFS;
  Status s = lib->LoadAndBind();
  if (s.IsOk()) {
    *result = lib;
    return s;
  } else {
    delete lib;
    return s;
  }
}

Status LibHDFS::LoadSymbols(void* dl) {
  #define LoadSymbol(sym) PS_CHECK_STATUS(BindFunc(dl, #sym, &sym));
  LoadSymbol(hdfsBuilderConnect);
  LoadSymbol(hdfsNewBuilder);
  LoadSymbol(hdfsBuilderSetNameNode);
  LoadSymbol(hdfsConfGetStr);
  LoadSymbol(hdfsBuilderSetKerbTicketCachePath);
  LoadSymbol(hdfsCloseFile);
  LoadSymbol(hdfsRead);
  LoadSymbol(hdfsPread);
  LoadSymbol(hdfsWrite);
  LoadSymbol(hdfsHFlush);
  LoadSymbol(hdfsHSync);
  LoadSymbol(hdfsOpenFile);
  LoadSymbol(hdfsExists);
  LoadSymbol(hdfsListDirectory);
  LoadSymbol(hdfsFreeFileInfo);
  LoadSymbol(hdfsDelete);
  LoadSymbol(hdfsCreateDirectory);
  LoadSymbol(hdfsGetPathInfo);
  LoadSymbol(hdfsRename);
  #undef LoadSymbol
  return Status::Ok();
}

Status LibHDFS::LoadAndBind() {
  const char* hdfs_root = getenv("HADOOP_HDFS_HOME");
  if (hdfs_root == nullptr) {
    return Status::NotFound("HADOOP_HDFS_HOME is not set");
  }
  const std::string& libhdfs = std::string(hdfs_root) + "/lib/native/libhdfs.so";
  void* dl = dlopen(libhdfs.c_str(), RTLD_LAZY);
  if (dl == nullptr) {
    return Status::NotFound("cannot find $HADOOP_HDFS_HOME/lib/native/libhdfs.so");
  }
  return LoadSymbols(dl);
}

}
}

