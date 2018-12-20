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

#ifndef XDL_CORE_OPS_PS_OPS_H_
#define XDL_CORE_OPS_PS_OPS_H_

#define DEFINE_OP_SINGLE_ARG(...) __VA_ARGS__

#define DEFINE_OP_SINGLE(TYPE, STMTS)       \
namespace define_op_##TYPE {               \
    typedef TYPE T;                         \
    STMTS;                                  \
}

#define DEFINE_INT_OP(STMTS)                                  \
namespace {                                                   \
  DEFINE_OP_SINGLE(int8_t, DEFINE_OP_SINGLE_ARG(STMTS))           \
  DEFINE_OP_SINGLE(int16_t, DEFINE_OP_SINGLE_ARG(STMTS))          \
  DEFINE_OP_SINGLE(int32_t, DEFINE_OP_SINGLE_ARG(STMTS))          \
  DEFINE_OP_SINGLE(int64_t, DEFINE_OP_SINGLE_ARG(STMTS))            \
}

#define DEFINE_ALL_TYPE_OP(STMTS)                             \
namespace {                                                   \
  DEFINE_OP_SINGLE(int8_t, DEFINE_OP_SINGLE_ARG(STMTS))           \
  DEFINE_OP_SINGLE(int16_t, DEFINE_OP_SINGLE_ARG(STMTS))          \
  DEFINE_OP_SINGLE(int32_t, DEFINE_OP_SINGLE_ARG(STMTS))          \
  DEFINE_OP_SINGLE(int64_t, DEFINE_OP_SINGLE_ARG(STMTS))            \
  DEFINE_OP_SINGLE(float, DEFINE_OP_SINGLE_ARG(STMTS))            \
  DEFINE_OP_SINGLE(double, DEFINE_OP_SINGLE_ARG(STMTS))           \
}

#endif // XDL_CORE_OPS_PS_OPS_H_
