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

#include "xdl/data_io/fs/file_system_kafka.h"

#include <memory>
#include <omp.h>
#include <stdlib.h>

#include "xdl/core/utils/logging.h"

namespace {
const char* kNameServiceKey = "NAME_SERVICE";
const char* kSeparator = ":";
int kTimeout = 10000; /* Timeout for Kafka client is 10 seconds */
}  // namespace 

namespace xdl {
namespace io {

/// Kafka IO ant
class IOAntKafka: public IOAnt {
 public:
  IOAntKafka(RdKafka::Topic *topic, RdKafka::Consumer *consumer, int partition) : topic_(topic), consumer_(consumer), partition_(partition), offset_(0) { }
  ~IOAntKafka() { 
    consumer_->stop(topic_, partition_);
    delete topic_;
    delete consumer_;
  }

  /*!\brief read data */
  virtual ssize_t Read(char *data, size_t len) override {
    std::unique_ptr<RdKafka::Message> msg(consumer_->consume(topic_, partition_, kTimeout));
    switch (msg->err()) {
      case RdKafka::ERR__TIMED_OUT:
        XDL_LOG(WARNING) << "Kafka consume time out";
        return 0;
      case RdKafka::ERR__PARTITION_EOF:
        XDL_LOG(WARNING) << "Kafka consume reach end of partition";
        return 0;
      case RdKafka::ERR_NO_ERROR:
        XDL_LOG(DEBUG) << "Kafka consume read " << msg->len();
        break;

      default:
        XDL_LOG(ERROR) << "Kafka consume fail! [" << topic_ << "," << partition_ << "]";
        return 0;
    }
    memcpy(data, msg->payload(), msg->len());
    len = msg->len();
    return len;
  }

  /*!\brief write data */
  virtual ssize_t Write(const char *data, size_t len) override {
    return -1;
  }

  /*!\brief seek to offset */
  virtual off_t Seek(off_t offset) override {
    offset_ = offset;
    consumer_->stop(topic_, partition_);
    RdKafka::ErrorCode err = consumer_->start(topic_, partition_, offset_);
    if (err != RdKafka::ERR_NO_ERROR) {
      XDL_LOG(ERROR) << "Failed to start consumer for offset " << offset_;
      return -1;
    }
    return offset_;
  }

 protected:
  RdKafka::Topic *topic_;
  RdKafka::Consumer *consumer_;
  int partition_;
  off_t offset_;
};

IOAnt *FileSystemKafka::GetAnt(const char *path, char mode) {
  std::string errstr;
  RdKafka::Conf *conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  if (!conf) {
    XDL_LOG(ERROR) << "Failed to create Kafka conf for " << namenode_;
    return nullptr;
  }
  conf->set("metadata.broker.list", namenode_, errstr);
  conf->set("group.id", "0", errstr);

  RdKafka::Consumer *consumer = RdKafka::Consumer::create(conf, errstr);
  if (!consumer) {
    XDL_LOG(ERROR) << "Failed to create Kafkaconsumer for " << namenode_
               << " [" << errstr << "]";
    return nullptr;
  }
  delete conf;

  RdKafka::Conf *tconf = RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC);
  if (!tconf) { 
    XDL_LOG(ERROR) << "Failed to create tconf";
    return nullptr;
  }

  /* Get topic string and partition from 'path' argument */
  size_t pos = std::string(path).find(kSeparator);
  if (pos == std::string::npos) {
    XDL_LOG(ERROR) << "Path of Kafka should be format of 'topic:partition'";
    return nullptr;
  }
  std::string topic_str(path, pos);
  int partition = atoi(path + pos + 1);

  RdKafka::Topic *topic = RdKafka::Topic::create(consumer, topic_str, tconf, errstr);
  if (!topic) {
    XDL_LOG(ERROR) << "Failed to create topic: "
               << " [" << errstr << "]";
    return nullptr;
  }
  delete tconf;

  RdKafka::ErrorCode err = consumer->start(topic, partition, 0);
  if (err != RdKafka::ERR_NO_ERROR) {
    XDL_LOG(ERROR) << "Failed to start consumer";
    return nullptr;
  }

  return new IOAntKafka(topic, consumer, partition);
}

bool FileSystemKafka::IsReg(const char *path) {
  size_t pos = std::string(path).find(kSeparator);
  if (pos == std::string::npos) {
    return false;
  }
  return true;
}

bool FileSystemKafka::IsDir(const char *path) {
  return false;
}

std::vector<std::string> FileSystemKafka::Dir(const char *path) {
  XDL_LOG(ERROR) << "Traverse of directory is not supported in Kafka";
  std::vector<std::string> paths;
  return paths;
}

void *FileSystemKafka::Open(const char *path, const char *mode) {
  XDL_LOG(ERROR) << "Open file is not supported in kafka";
  return nullptr;
}

size_t FileSystemKafka::Size(const char *path) {
  return size_t(-1);
}

/// FileSystemKafka Implementation
FileSystemKafka::~FileSystemKafka() { }

FileSystem *FileSystemKafka::Get(const char* namenode) {
  static std::shared_ptr<FileSystemKafka> inst(new FileSystemKafka(namenode));
  return inst.get();
}

FileSystemKafka::FileSystemKafka(const char* namenode) : namenode_(namenode) {
  if (namenode_.empty()) {
    namenode_.append(getenv(kNameServiceKey));
    XDL_LOG(DEBUG) << "nameservice=" << namenode_;
  }
}

}  // namespace io
}  // namespace xdl
