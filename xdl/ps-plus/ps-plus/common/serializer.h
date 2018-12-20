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

#ifndef PS_COMMON_SERIALIZER_H_
#define PS_COMMON_SERIALIZER_H_

#include <sys/uio.h>
#include <typeindex>
#include <functional>
#include <memory>

#include "status.h"
#include "plugin.h"
#include "data.h"
#include "memguard.h"

namespace ps {
namespace serializer {

struct Fragment {
  Fragment()
    : base(nullptr)
    , size(0) {
  }
  Fragment(char* buf, size_t len)
    : base(buf)
    , size(len) {
  }
  char* base;
  size_t size;
};

template<typename T>
constexpr size_t SerializerTypeHash() {
  std::hash<std::string> hash_fn;
  return hash_fn(std::string(typeid(T).name()));
}

template<typename Tbase>
class SerializerBase {
 public:
  using BaseType = Tbase;
  virtual ~SerializerBase() {}

  static ps::Status SerializeAny(Tbase* data, size_t* id, std::vector<Fragment>* iovecs, MemGuard& mem_guard) {
    if (data == nullptr) {
      return ps::Status::ArgumentError("Serializer should not accept nullptr");
    }
    SerializerBase* serializer = GetPlugin<SerializerBase, std::type_index>(std::type_index(typeid(*data)));
    if (serializer == nullptr) {
      return ps::Status::NotFound(std::string("Serializer not found for data type:") + typeid(*data).name());
    }
    return serializer->SerializeWrapper(data, id, iovecs, mem_guard);
  }
 protected:
  virtual ps::Status SerializeWrapper(Tbase* data, size_t* id, std::vector<Fragment>* iovecs, MemGuard& mem_guard) = 0;
};

template<typename Tbase, typename Tdata, typename Tdst = Tdata>
class Serializer : public SerializerBase<Tbase> {
 public:
  using DataType = Tdata;
  Serializer() : id_(SerializerTypeHash<Tdst>()) {}
 protected:
  virtual ps::Status SerializeWrapper(Tbase* data, size_t* id, std::vector<Fragment>* iovecs, MemGuard& mem_guard) {
    Tdata* new_data = dynamic_cast<Tdata*>(data);
    if (new_data == nullptr) {
      // This should not be avaliable. SerializeAny should promised data is a pointer to Tdata.
      return ps::Status::ArgumentError(std::string("Serializer Accept Type ") + typeid(Tdata).name() + " but got " + typeid(*data).name());
    }
    *id = id_;
    return Serialize(new_data, iovecs, mem_guard);
  }
  virtual ps::Status Serialize(Tdata* data, std::vector<Fragment>* iovecs, MemGuard& mem_guard) = 0;
  size_t id_;
};

template<typename Tbase>
class DeserializerBase {
 public:
  using BaseType = Tbase;
  virtual ~DeserializerBase() {}

  static ps::Status DeserializeAny(size_t id, Fragment* buffer, size_t offset, Tbase** result, size_t* len, MemGuard& mem_guard) {
    DeserializerBase* deserializer = GetPlugin<DeserializerBase, size_t>(id);
    if (deserializer == nullptr) {
      return ps::Status::NotFound(std::string("Deserializer not found for id:") + std::to_string(id));
    }
    return deserializer->DeserializeWrapper(id, buffer, offset, result, len, mem_guard);
  }
 protected:
  virtual ps::Status DeserializeWrapper(size_t id, Fragment* buffer, size_t offset, Tbase** result, size_t* len, MemGuard& mem_guard) = 0;
};

template<typename Tbase, typename Tdata>
class Deserializer : public DeserializerBase<Tbase> {
 public:
  using DataType = Tdata;
  Deserializer() : id_(SerializerTypeHash<DataType>()) {}
 protected:
  virtual ps::Status DeserializeWrapper(size_t id, Fragment* buffer, size_t offset, Tbase** result, size_t* len, MemGuard& mem_guard) {
    if (id != id_) {
      // This should not be avaliable. SerializeAny should promised data is a pointer to Tdata.
      return ps::Status::ArgumentError(std::string("DeSerializer Accept Type ") + std::to_string(SerializerTypeHash<DataType>()) + " but got " + std::to_string(id));
    }
    Tdata* result_new;
    PS_CHECK_STATUS(Deserialize(buffer, offset, &result_new, len, mem_guard));
    *result = result_new;
    return ps::Status::Ok();
  }
  virtual ps::Status Deserialize(Fragment* buffer, size_t offset, Tdata** result, size_t* len, MemGuard& mem_guard) = 0;
  size_t id_;
};

template<typename Tbase>
ps::Status SerializeAny(Tbase* data, size_t* id, std::vector<Fragment>* iovecs, MemGuard& mem_guard) {
  return SerializerBase<Tbase>::SerializeAny(data, id, iovecs, mem_guard);
}

template<typename Tbase>
ps::Status DeserializeAny(size_t id, Fragment* buffer, size_t offset, Tbase** result, size_t* len, MemGuard& mem_guard) {
  return DeserializerBase<Tbase>::DeserializeAny(id, buffer, offset, result, len, mem_guard);
}

} // namespace serializer
} // namespace ps

#define SERIALIZER_REGISTER(SERILIAZER_TYPE)                            \
  PLUGIN_REGISTER_TYPE(ps::serializer::SerializerBase<SERILIAZER_TYPE::BaseType>, std::type_index, std::type_index(typeid(SERILIAZER_TYPE::DataType)), SERILIAZER_TYPE)

#define DESERIALIZER_REGISTER(DESERILIAZER_TYPE)                        \
  PLUGIN_REGISTER_TYPE(ps::serializer::DeserializerBase<DESERILIAZER_TYPE::BaseType>, size_t, ps::serializer::SerializerTypeHash<DESERILIAZER_TYPE::DataType>(), DESERILIAZER_TYPE)

#endif // PS_COMMON_SERIALIZER_H_

