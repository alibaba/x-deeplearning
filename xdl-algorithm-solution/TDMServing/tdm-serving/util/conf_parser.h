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

#ifndef TDM_SERVING_UTIL_CONF_PARSER_H_
#define TDM_SERVING_UTIL_CONF_PARSER_H_

#include <tr1/unordered_map>
#include <string>
#include <cstring>
#include <map>
#include "util/str_util.h"

namespace tdm_serving {
namespace util {

// conf kv define
class ConfKv {
 public:
  ConfKv(const std::string& key, const std::string& value);
  ~ConfKv();
  void Clear();
  const std::string& GetKey() const;
  const std::string& GetValue() const;

 private:
  std::string* key_;
  std::string* value_;

  DISALLOW_COPY_AND_ASSIGN(ConfKv);
};

// conf section define
class ConfSection {
 public:
  explicit ConfSection(const std::string& section);
  ~ConfSection();
  void Clear();
  const std::string& GetSectionName() const;
  const std::vector<ConfKv*>& GetAllKvs() const;
  const ConfKv* GetKv(const std::string& key) const;
  bool AddKv(const std::string& key, const std::string& value);

 protected:
  std::string section_;
  std::vector<ConfKv*> kv_vec_;
  std::tr1::unordered_map<std::string, ConfKv*> kv_map_;

 private:
  DISALLOW_COPY_AND_ASSIGN(ConfSection);
};

// conf parser define
class ConfParser {
 public:
  ConfParser();
  virtual ~ConfParser();

  // Load conf file
  bool Init(const std::string& kv_conf);

  // Clear
  void Clear();

  // Get value by section and key, if not exists return false
  template<typename Type>
  bool GetValue(const std::string& section,
      const std::string& key, Type* value) const;

  // Get value by section and key, if not exists set to default
  template<typename Type>
  void GetValue(
      const std::string& section, const std::string& key,
      const Type& default_value, Type* value) const;

  const std::vector<ConfSection*>& GetAllConfSection() const;
  const ConfSection* GetConfSection(const std::string& section) const;

  ConfSection* AddConfSection(const std::string& section);

 protected:
  std::vector<ConfSection*> conf_section_vec_;
  std::tr1::unordered_map<std::string, ConfSection*> conf_section_map_;

 private:
  DISALLOW_COPY_AND_ASSIGN(ConfParser);
};

// Get value by section and key, if not exists return false
template<typename Type>
bool ConfParser::GetValue(const std::string& section,
    const std::string& key, Type* value) const {
  if (value == NULL) {
    return false;
  }

  const ConfSection* conf_section = GetConfSection(section);
  if (conf_section == NULL) {
    return false;
  }

  const ConfKv* conf_kv = conf_section->GetKv(key);
  if (conf_kv == NULL) {
    return false;
  }

  if (!StrUtil::StrConvert<Type>(conf_kv->GetValue().c_str(), value)) {
    return false;
  }

  return true;
}

template<typename Type>
void ConfParser::GetValue(
    const std::string& section, const std::string& key,
    const Type& default_value, Type* value) const {
  if (value == NULL) {
    return;
  }

  if (!GetValue<Type>(section, key, value)) {
    *value = default_value;
  }
}

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_CONF_PARSER_H_
