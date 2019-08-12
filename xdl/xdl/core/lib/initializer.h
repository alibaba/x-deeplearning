/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef XDL_CORE_LIB_INITIALIZER_H_
#define XDL_CORE_LIB_INITIALIZER_H_

namespace xdl {
namespace common {

class Initializer {
 public:
  typedef void (*InitializerFunc)();
  explicit Initializer(InitializerFunc func) { func(); }
};

#ifndef REGISTER_INITIALIZER
#define REGISTER_INITIALIZER(name, body) \
    static void _xdl_init_##name() { body; } \
    static xdl::common::Initializer __xdl_init_##name(_xdl_init_##name)
#endif

}  // namespace common
}  // namespace xdl

#endif  // XDL_CORE_LIB_INITIALIZER_H_
