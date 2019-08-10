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

#ifndef XDL_IO_CONSTANT_H_
#define XDL_IO_CONSTANT_H_

#include <string>
#include <limits>

namespace xdl {
namespace io {

static const void *END = (void *)(-1);

/// The defined filesystem name
/*
static const char *kHdfsName = "hdfs";
static const char *kLocalName = "local";
static const char *kSwiftName = "swift";
*/

/*! \brief type of fs */
enum FSType {
  /*! \brief the Local */
  kLocal = 0x01,
  /*! \brief the hdfs */
  kHdfs = 0x02,
  /*! \brief the swif */
  kSwift = 0x03,
  /*! \brief the odps */
  kOdps = 0x04,
  /*! \brief the kafka */
  kKafka = 0x05,
};

enum ParserType {
  kPB = 0x01,
  kTxt = 0x02,
  kTfRnn = 0x03,
  kV4 = 0x04,
  kSPB = 0x05,
};

enum ZType {
  kRaw = 0x00,
  kZLib = 0x01,
  kGZip = 0x02,
};

const size_t MAX_END_TIME = std::numeric_limits<size_t>::max() / 8;


}  // namespace io
}  // namespace xdl

#endif // XDL_IO_CONSTANT_H_
