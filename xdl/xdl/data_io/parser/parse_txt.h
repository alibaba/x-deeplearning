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


#ifndef XDL_CORE_IO_PARSE_TXT_H_
#define XDL_CORE_IO_PARSE_TXT_H_

#include "xdl/data_io/parser/parser.h"

namespace xdl {
namespace io {

/* each field seperated by |
 * sample id: string
 * group id:  string
 * kvs:       sf1@k1:v1,k2:v2;sf2@k3,k4
 * dense:     df1@v1,v2;df2@v3
 * label:     l1,l2 
 * ts:        timestamp
 */

const size_t MAX_NUM_SEG = 6;
const size_t MAX_NUM_FEA = 1024;
const size_t MAX_NUM_VAL = 1024;
const size_t MAX_NUM_LAB = 16;

const size_t MAX_NUM_SAMPLE_OF_GROUP = 8192;

const char kSEG = '|';
const char kNAM = '@';
const char kFEA = ';';
const char kVAL = ',';
const char kKEY = ':';

class ParseTxt : public Parse {
 public:
  ParseTxt(const Schema *schema):Parse(schema) {}
  virtual ~ParseTxt() {}

  virtual SGroup *Run(const char *str, size_t len) override;
  virtual ssize_t GetSize(const char *str, size_t len) override;

  using Closure = std::function<void(const char *str, size_t len, int i)>;
  bool OnLabel(Label *l, const char *str, size_t len);
  bool OnFeatureLine(FeatureLine *fl, const char *str, size_t len, FeatureType type);
  bool OnFeature(Feature *f, const char *str, size_t len, FeatureType type);
  static bool OnDense(FeatureValue *v, const char *s, size_t n);
  static bool OnSparse(FeatureValue *v, const char *s, size_t n);

  static int Tokenize(const char *ptrs[], size_t lens[], const char *str, size_t len, char c, size_t max_count);
  static int Tokenize(const char *str, size_t len, char c, size_t max_count, Closure closure);

 private:
  std::set<std::string> main_features_;
  std::set<std::string> compact_features_;

  std::string last_group_key_;
  SGroup *last_sgroup_ = nullptr;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_CORE_IO_PARSE_TXT_H_