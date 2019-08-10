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


#ifndef XDL_CORE_IO_PARSE_V4_H_
#define XDL_CORE_IO_PARSE_V4_H_

#include "xdl/data_io/parser/parser.h"
#include "xdl/proto/sample_v4.pb.h"

namespace xdl {
namespace io {

class ParseV4 : public Parse {
 public:
  ParseV4(const Schema *schema):Parse(schema) {}
  virtual ~ParseV4() {}

  virtual bool InitMeta(const std::string &contents) override;
  virtual SGroup *Run(const char *str, size_t len) override;
  virtual ssize_t GetSize(const char *str, size_t len) override;

 protected:
  v4::SampleMeta meta_;
  std::vector<std::string> ncomm_;
  std::vector<std::string> comm_;

  bool OnLabel(const v4::DataBlock &block, SampleGroup *sg);
  bool OnSKey(const v4::DataBlock &block, SampleGroup *sg);
  bool OnTable(const v4::DataBlock &block, FeatureTable *tab);
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_PARSE_V4_H_