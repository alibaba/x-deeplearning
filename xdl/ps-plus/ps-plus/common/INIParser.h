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

#ifndef PS_COMMON_INI_PARSER_H_
#define PS_COMMON_INI_PARSER_H_

#include <string>
#include <map>

namespace ps
{

typedef std::map<std::string, std::string> stringMap; // ±£´ækey => value
typedef std::map<std::string, stringMap> sectionMap;  // ±£´æsection => key/value

class INIParser
{
 public:
  INIParser();
  INIParser(const char *profile);
  INIParser(const std::string &profile);
  virtual ~INIParser();

  void load(const char *profile);
  void load(const std::string &profile);
  const std::string &get_string(const std::string &section, const std::string &key, const std::string &def)const;
  const char *get_string(const char *section, const char *key, const char *def = NULL)const;
  int get_int(const std::string &section, const std::string &key, int def = 0)const;
  int get_int(const char *section, const char *key, int def = 0)const;
  unsigned get_unsigned(const std::string &section, const std::string &key, unsigned def = 0)const;
  unsigned get_unsigned(const char *section, const char *key, unsigned def = 0)const;
  bool get_bool(const std::string &section, const std::string &key, bool def = false)const;
  bool get_bool(const char *section, const char *key, bool def = false)const;
  std::string &get_section(const std::string &section);
  const char *get_section(const char *section);
  void dump()const;
 private:
  sectionMap m_ini;
  stringMap m_sections;
  std::string m_none;
  std::string NOTHING;

  bool find_section(const char *p, std::string &section, std::string &sq_section, std::string &sec_body);
  const char *find_pair(const char *p, std::string &section);
  void append_sq_section(std::string &sq_section, std::string &sec_body, const char *p, int len=-1);
};

}

#endif
