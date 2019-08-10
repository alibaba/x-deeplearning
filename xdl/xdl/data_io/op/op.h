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


#ifndef XDL_CORE_IO_OP_H_
#define XDL_CORE_IO_OP_H_

#include "xdl/proto/sample.pb.h"

#include <string>
#include <map>

#include "xdl/core/lib/singleton.h"
#include "xdl/data_io/schema.h"

namespace xdl {
namespace io {

class Operator {
 public:
  Operator() {}
  virtual ~Operator() {}

  inline bool set_schema(Schema *schema) {
    if (schema_ != nullptr)  return false;
    schema_ = schema;
    return true;
  }

  virtual bool Init(const std::map<std::string, std::string> &params) {
    params_ = params;
    return true;
  }
  virtual bool Run(SampleGroup *sample_group) = 0;
  virtual std::map<std::string, std::string> URun(const std::map<std::string, std::string> &params) {
    std::map<std::string, std::string> out;
    return out;
  }
 protected:
  std::map<std::string, std::string> params_;
  const Schema *schema_ = nullptr;
};

class IOPRegistry: public Singleton<IOPRegistry> {
  using Creator = std::function<Operator *()>;

 public:
  Operator *GetOP(const std::string &key) {
    auto it = creators_.find(key);
    if (it == creators_.end()) {
      return nullptr;
    }
    return it->second();
  }

  bool AddCreator(const std::string &key, Creator c) {
    creators_.insert(make_pair(key, c));
    return true;
  }

 private:
  std::map<const std::string, Creator> creators_;
};

}  // namespace io
}  // namespace xdl

#define XDL_REGISTER_IOP(type)                    \
static auto p = xdl::io::IOPRegistry::Get()->AddCreator(#type, []()->xdl::io::Operator *{return new type();});


#endif  // XDL_CORE_IO_OP_H_