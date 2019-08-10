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

#ifndef XDL_CORE_IO_FILTEROP_H_
#define XDL_CORE_IO_FILTEROP_H_

#include "xdl/data_io/op/op.h"

#include <atomic>
#include <string>
#include <set>
#include <vector>

namespace xdl {
namespace io {

class FilterOP : public Operator {
 public:
  FilterOP() {filtered_ = 0; passed_ = 0;}
  virtual ~FilterOP() {}

  virtual bool Init(const std::map<std::string, std::string> &params) override;
  virtual bool Run(SampleGroup *sample_group) override;
  virtual std::map<std::string, std::string> URun(const std::map<std::string, std::string> &params) override;

 private:
  std::set<std::string> del_skeys_;
  std::atomic<unsigned long> filtered_;
  std::atomic<unsigned long> passed_;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_FILTER_H_
