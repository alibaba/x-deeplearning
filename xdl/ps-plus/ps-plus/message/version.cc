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

#include "version.h"

#include <limits>
#include <random>

namespace ps {

Version NewRandomVersion() {
  static std::random_device engine;
  static std::uniform_int_distribution<Version> dist(std::numeric_limits<Version>::min());
  Version v = dist(engine);
  return v == 0 ? 1 : v;
}

}
