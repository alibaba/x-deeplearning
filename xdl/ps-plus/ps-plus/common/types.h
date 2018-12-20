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

#ifndef PS_COMMON_TYPES_H
#define PS_COMMON_TYPES_H

#include <cstdint>
#include <string>

namespace ps {

namespace types {
enum DataType : int32_t {
    kInt8 = 0,
    kInt16 = 1,
    kInt32 = 2,
    kInt64 = 3,
    kFloat = 4,
    kDouble = 5
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

#undef MATCH_TYPE_AND_ENUM

#define SINGLE_ARG(...) __VA_ARGS__

#define CASE(TYPE, STMTS)                                     \
    case ::ps::DataTypeToEnum<TYPE>::value: {                 \
        typedef TYPE T;                                       \
        STMTS;                                                \
        break;                                                \
    }

#define CASES(TYPE_ENUM, STMTS)                 \
    switch (TYPE_ENUM) {                        \
        CASE(int8_t, SINGLE_ARG(STMTS))         \
        CASE(int16_t, SINGLE_ARG(STMTS))        \
        CASE(int32_t, SINGLE_ARG(STMTS))        \
        CASE(int64_t, SINGLE_ARG(STMTS))        \
        CASE(float, SINGLE_ARG(STMTS))          \
        CASE(double, SINGLE_ARG(STMTS))         \
    }

inline size_t SizeOfType(DataType type) {
  CASES(type, return sizeof(T));
  return 0;
}

} //ps

#endif

