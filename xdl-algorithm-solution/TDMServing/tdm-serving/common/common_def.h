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

#ifndef TDM_SERVING_COMMON_COMMON_DEF_H_
#define TDM_SERVING_COMMON_COMMON_DEF_H_

#include <inttypes.h>
#include <string>

namespace tdm_serving {

extern const std::string kMetaSection;

extern const std::string kConfigEnable;
extern const std::string kConfigIndexType;
extern const std::string kConfigIndexPath;
extern const std::string kConfigIndexModelName;
extern const std::string kConfigIndexFilterName;
extern const std::string kConfigIndexBuildOmp;
extern const std::string kConfigModelType;
extern const std::string kConfigModelPath;
extern const std::string kConfigFilterType;

extern const std::string kVersionFile;
extern const std::string kIndexVersionTag;
extern const std::string kModelVersionTag;

extern const uint32_t kIndexInstanceNum;
extern const uint32_t kModelInstanceNum;

extern const uint32_t ktObjectPoolInitSize;


#define COPY_CONSTRUCTOR(T)                  \
  T(const T&);                               \
  T& operator=(const T&);

#define DISALLOW_COPY_AND_ASSIGN(T)          \
  COPY_CONSTRUCTOR(T)

#define DELETE_AND_SET_NULL(x) do {          \
  if (x != NULL) {                           \
    delete x;                                \
    x = NULL;                                \
  }                                          \
} while (0)

#define DELETE_ARRAY(x) do {                 \
  if (x != NULL) {                           \
    delete [] x;                             \
    x = NULL;                                \
  }                                          \
} while (0)

#define ARRAY_DELETE_AND_SET_NULL(x) delete [] x; x = NULL

}  // namespace tdm_serving

#endif  // TDM_SERVING_COMMON_COMMON_DEF_H_
