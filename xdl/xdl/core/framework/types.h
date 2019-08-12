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

#ifndef XDL_CORE_FRAMEWORK_TYPES_H_
#define XDL_CORE_FRAMEWORK_TYPES_H_

#include <cstdint>
#include <string>

#include "xdl/core/proto/graph_def.pb.h"
#include "xdl/core/utils/logging.h"

namespace xdl {

namespace types {
enum DataType : int32_t {
    kInt8 = proto::DataType::kInt8,
    kInt16 = proto::DataType::kInt16,
    kInt32 = proto::DataType::kInt32,
    kInt64 = proto::DataType::kInt64,
    kFloat = proto::DataType::kFloat,
    kDouble = proto::DataType::kDouble,
    kBool = proto::DataType::kBool
};
}

using types::DataType;

/**
 * template used to convert between enum and type
 */
template <class T>
struct DataTypeToEnum {};

template <DataType VALUE>
struct EnumToDataType {};

#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                   \
    template <>                                           \
    struct DataTypeToEnum<TYPE> {                         \
        static DataType v() { return DataType::ENUM; }    \
        static constexpr DataType value = DataType::ENUM; \
    };                                                    \
    template <>                                           \
    struct EnumToDataType<DataType::ENUM> {               \
        typedef TYPE Type;                                \
    }

MATCH_TYPE_AND_ENUM(int8_t, kInt8);
MATCH_TYPE_AND_ENUM(int16_t, kInt16);
MATCH_TYPE_AND_ENUM(int32_t, kInt32);
MATCH_TYPE_AND_ENUM(int64_t, kInt64);
MATCH_TYPE_AND_ENUM(float, kFloat);
MATCH_TYPE_AND_ENUM(double, kDouble);
MATCH_TYPE_AND_ENUM(bool, kBool);

#undef MATCH_TYPE_AND_ENUM

#define XDL_TYPE_SINGLE_ARG(...) __VA_ARGS__

#define XDL_TYPE_CASE(TYPE, STMTS)                            \
    case ::xdl::DataTypeToEnum<TYPE>::value: {                \
        typedef TYPE T;                                       \
        STMTS;                                                \
        break;                                                \
    }

#define XDL_TYPE_CASES(TYPE_ENUM, STMTS)                          \
    switch (TYPE_ENUM) {                                          \
        XDL_TYPE_CASE(int8_t, XDL_TYPE_SINGLE_ARG(STMTS))         \
        XDL_TYPE_CASE(int16_t, XDL_TYPE_SINGLE_ARG(STMTS))        \
        XDL_TYPE_CASE(int32_t, XDL_TYPE_SINGLE_ARG(STMTS))        \
        XDL_TYPE_CASE(int64_t, XDL_TYPE_SINGLE_ARG(STMTS))        \
        XDL_TYPE_CASE(float, XDL_TYPE_SINGLE_ARG(STMTS))          \
        XDL_TYPE_CASE(double, XDL_TYPE_SINGLE_ARG(STMTS))         \
        XDL_TYPE_CASE(bool, XDL_TYPE_SINGLE_ARG(STMTS))           \
        default: XDL_CHECK(false) << "type error";                    \
    }

inline size_t SizeOfType(DataType type) {
  size_t ret;
  XDL_TYPE_CASES(type, ret = sizeof(T));
  return ret;
}

}  // namespace xdl

#endif  // XDL_CORE_FRAMEWORK_TYPES_H_

