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

#ifndef XDL_CORE_LIB_BASE64_H_
#define XDL_CORE_LIB_BASE64_H_

#include <string>

namespace xdl {
  int32_t Base64Encode(unsigned char *dst, const uint32_t &dsize,
                       const unsigned char *src, const uint32_t &size);

  int32_t Base64Encode(std::string *dst, const std::string &src);

  int32_t Base64Decode(unsigned char *dst, const uint32_t &dsize,
                       const unsigned char *src, const uint32_t &size);

  int32_t Base64Decode(std::string *dst, const std::string &src);
}  // namespace xdl

#endif  // XDL_CORE_LIB_BASE64_H_
