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

#ifndef XDL_IO_FS_FILE_SYSTEM_KAFKA_H_
#define XDL_IO_FS_FILE_SYSTEM_KAFKA_H_

#include <stdio.h>
#include <unordered_map>

#include "xdl/data_io/fs/file_system.h"
#include "third_party/librdkafka/src-cpp/rdkafkacpp.h"

namespace xdl {
namespace io {

class FileSystemKafka : public FileSystem {
 public:
  /*! \brief virtual destructor */
  virtual ~FileSystemKafka();
  virtual IOAnt *GetAnt(const char *path, char mode = 'r') override;

  virtual bool IsDir(const char *path) override;
  virtual bool IsReg(const char *path) override;
  virtual std::vector<std::string> Dir(const char *path) override;
  virtual void *Open(const char *path, const char *mode) override;
  virtual size_t Size(const char *path) override;

  /*!
   * \brief Get a singleton of KafkaFileSystem
   */
  static FileSystem *Get(const char* namenode = "localhost:9092");
  
 protected:
  FileSystemKafka(const char* namenode);

  /*! the host:port of kafka service */
  std::string namenode_;
  /*! the consumer handle of Kafka */
  RdKafka::Consumer *consumer_;
  RdKafka::Topic *topic_;
  int partition_;
};

std::string KafkaRead(const std::string& name);

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_FS_FILE_SYSTEM_KAFKA_H_
