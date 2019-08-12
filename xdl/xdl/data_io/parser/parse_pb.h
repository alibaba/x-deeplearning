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


#ifndef XDL_CORE_IO_PARSE_PB_H_
#define XDL_CORE_IO_PARSE_PB_H_

#include "xdl/data_io/parser/parser.h"

namespace xdl {
namespace io {

class ParsePB : public Parse {
 public:
  ParsePB(const Schema *schema):Parse(schema) {}
  virtual ~ParsePB() {}

  virtual SGroup *Run(const char *str, size_t len) override;
  virtual ssize_t GetSize(const char *str, size_t len) override;

 protected:
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_PARSE_PB_H_