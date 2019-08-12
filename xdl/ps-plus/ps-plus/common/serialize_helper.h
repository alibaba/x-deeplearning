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

#ifndef PS_COMMON_SERIALIZE_HELPER_H_
#define PS_COMMON_SERIALIZE_HELPER_H_

#include <cstring>
#include <limits>
#include "ps-plus/common/logging.h"

#define private public

#include "status.h"
#include "data.h"
#include "tensor.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/initializer/normal_initializer.h"
#include "ps-plus/common/initializer/truncated_normal_initializer.h"
#include "ps-plus/common/initializer/variance_scaling_initializer.h"
#include "ps-plus/common/initializer/uniform_unit_scaling_initializer.h"
#include "ps-plus/common/initializer/orthogonal_initializer.h"
#include "ps-plus/common/hashmap.h"

#undef private

#include "plugin.h"
#include "serializer.h"

namespace ps {
namespace serializer {

// helper class for serialize
class SerializeHelper {
 public:
  template <typename T>
  static ps::Status Serialize(const T* data, 
                              std::vector<Fragment>* bufs, 
                              MemGuard& mem_guard) {
    Fragment frag(reinterpret_cast<char*>(mem_guard.AllocateElement<T>(*data)), sizeof(T));
    bufs->push_back(frag);
    return ps::Status::Ok();
  }

  template <typename T>  
  static ps::Status SerializeVec(const std::vector<T>* data, 
                              std::vector<Fragment>* bufs, 
                              MemGuard& mem_guard) {
    size_t size = data->size();
    PS_CHECK_STATUS(SerializeHelper::Serialize<size_t>(&size, bufs, mem_guard));
    for (size_t i = 0; i < size; i++) {
      PS_CHECK_STATUS(SerializeHelper::Serialize<T>(&(*data)[i], bufs, mem_guard));
    }
    return ps::Status::Ok();
  }

  template <typename T>
  static ps::Status Deserialize(const char* buf, 
                                T* data,
                                size_t* len, 
                                MemGuard& mem_guard) {
    *data = *(reinterpret_cast<const T*>(buf));
    *len = sizeof(T);
    return ps::Status::Ok();
  }

  template <typename T>
  static ps::Status DeserializeVec(const char* buf, 
                                   std::vector<T>* data,
                                   size_t* len, 
                                   MemGuard& mem_guard) {
    size_t size;
    size_t field_len;
    PS_CHECK_STATUS(Deserialize<size_t>(buf, &size, &field_len, mem_guard));
    size_t offset = 0;
    for (size_t i = 0; i < size; i++) {
      offset += field_len;
      T t;
      PS_CHECK_STATUS(Deserialize<T>(buf + offset, &t, &field_len, mem_guard));
      data->push_back(std::move(t));
    }
    *len = offset + field_len;
    return ps::Status::Ok();
  }
};

// specification for SerializeHelper
template <>  
ps::Status SerializeHelper::Serialize<std::string>(const std::string* data, 
                                                   std::vector<Fragment>* bufs,
                                                   MemGuard& mem_guard) {
  Fragment size(reinterpret_cast<char*>(mem_guard.AllocateElement<size_t>(data->size())), 
                sizeof(size_t));
  bufs->push_back(size);
  Fragment buf(const_cast<char*>(data->c_str()), data->size());
  bufs->push_back(buf);
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<std::string>(const char* buf, 
                                                     std::string* data, 
                                                     size_t* len,
                                                     MemGuard& mem_guard) {
  char* buf_start = const_cast<char*>(buf);
  *len = *(reinterpret_cast<size_t*>(buf_start));
  data->assign(buf_start + sizeof(size_t), *len);
  *len += sizeof(size_t);
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<std::vector<std::string> >(
    const std::vector<std::string>* data, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  size_t buffer_size = sizeof(size_t);
  for (size_t i = 0; i < data->size(); i++) {
    buffer_size += sizeof(size_t) + data->at(i).size();
  }
  char* buffer = mem_guard.AllocateBuffer(buffer_size);
  *(size_t*)buffer = data->size();
  size_t offset = sizeof(size_t);
  for (size_t i = 0; i < data->size(); i++) {
    *(size_t*)(buffer+offset) = data->at(i).size();
    offset += sizeof(size_t);
    memcpy(buffer + offset, data->at(i).c_str(), data->at(i).size());
    offset += data->at(i).size();
  }
  bufs->push_back(Fragment{.base=buffer, .size=buffer_size});
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<std::vector<std::string> >(
    const char* buf, 
    std::vector<std::string>* data, 
    size_t* len,
    MemGuard& mem_guard) {
  return SerializeHelper::DeserializeVec(buf, data, len, mem_guard);
}

template <>  
ps::Status SerializeHelper::Serialize<std::vector<bool> >(
    const std::vector<bool>* data, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  size_t buffer_size = sizeof(size_t) + data->size();
  char* buffer = mem_guard.AllocateBuffer(buffer_size);  
  *(size_t*)buffer = data->size();
  size_t offset = sizeof(size_t);
  for (size_t i = 0; i < data->size(); i++) {
    buffer[offset++] = (*data)[i];
  }
  Fragment buf(buffer, buffer_size);
  bufs->push_back(std::move(buf));
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<std::vector<bool> >(
    const char* buf, 
    std::vector<bool>* data, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t size;
  size_t field_len;
  PS_CHECK_STATUS(Deserialize<size_t>(buf, &size, &field_len, mem_guard));
  size_t offset = 0;
  for (size_t i = 0; i < size; i++) {
    offset += field_len;
    bool t;
    PS_CHECK_STATUS(Deserialize<bool>(buf + offset, &t, &field_len, mem_guard));
    data->push_back(t);
  }
  *len = offset + field_len;
  return ps::Status::Ok();  
}

template <>  
ps::Status SerializeHelper::Serialize<std::vector<double> >(
    const std::vector<double>* data, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  size_t buffer_size = sizeof(size_t);
  buffer_size += data->size() * sizeof(double);
  char* buffer = mem_guard.AllocateBuffer(buffer_size);  
  *(size_t*)buffer = data->size();
  size_t offset = sizeof(size_t);
  for (size_t i = 0; i < data->size(); i++) {
    *(double*)(buffer + offset) = data->at(i);
    offset += sizeof(double);
  }
  bufs->push_back(Fragment{.base=buffer, .size=buffer_size});  
  return Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<std::vector<float> >(
    const std::vector<float>* data, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  size_t buffer_size = sizeof(size_t);
  buffer_size += data->size() * sizeof(float);
  char* buffer = mem_guard.AllocateBuffer(buffer_size);  
  *(size_t*)buffer = data->size();
  size_t offset = sizeof(size_t);
  for (size_t i = 0; i < data->size(); i++) {
    *(float*)(buffer + offset) = data->at(i);
    offset += sizeof(float);
  }
  bufs->push_back(Fragment{.base=buffer, .size=buffer_size});  
  return Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<std::vector<int64_t> >(
    const std::vector<int64_t>* data, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  size_t buffer_size = sizeof(size_t);
  buffer_size += data->size() * sizeof(int64_t);
  char* buffer = mem_guard.AllocateBuffer(buffer_size);  
  *(size_t*)buffer = data->size();
  size_t offset = sizeof(size_t);
  for (size_t i = 0; i < data->size(); i++) {
    *(int64_t*)(buffer + offset) = data->at(i);
    offset += sizeof(int64_t);
  }
  bufs->push_back(Fragment{.base=buffer, .size=buffer_size});  
  return Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<std::vector<int64_t> >(
    const char* buf, 
    std::vector<int64_t>* data, 
    size_t* len,
    MemGuard& mem_guard) {
  return SerializeHelper::DeserializeVec<int64_t>(buf, data, len, mem_guard);
}
template <>  
ps::Status SerializeHelper::Serialize<std::vector<int> >(
    const std::vector<int>* data, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  size_t buffer_size = sizeof(size_t);
  buffer_size += data->size() * sizeof(int);
  char* buffer = mem_guard.AllocateBuffer(buffer_size);  
  *(size_t*)buffer = data->size();
  size_t offset = sizeof(size_t);
  for (size_t i = 0; i < data->size(); i++) {
    *(int*)(buffer + offset) = data->at(i);
    offset += sizeof(int);
  }
  bufs->push_back(Fragment{.base=buffer, .size=buffer_size});  
  return Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<std::vector<int> >(
    const char* buf, 
    std::vector<int>* data, 
    size_t* len,
    MemGuard& mem_guard) {
  return SerializeHelper::DeserializeVec<int>(buf, data, len, mem_guard);
}


template <>
ps::Status SerializeHelper::Deserialize<std::vector<double> >(
    const char* buf, 
    std::vector<double>* data, 
    size_t* len,
    MemGuard& mem_guard) {
  return SerializeHelper::DeserializeVec<double>(buf, data, len, mem_guard);
}

template <>
ps::Status SerializeHelper::Deserialize<std::vector<float> >(
    const char* buf, 
    std::vector<float>* data, 
    size_t* len,
    MemGuard& mem_guard) {
  return SerializeHelper::DeserializeVec<float>(buf, data, len, mem_guard);
}

template <>  
ps::Status SerializeHelper::Serialize<ps::Status>(const ps::Status* st, 
                                                  std::vector<Fragment>* bufs,
                                                  MemGuard& mem_guard) {
  if (st->state_ != nullptr) {
    Serialize<int32_t>(reinterpret_cast<int32_t*>(&(st->state_->code)), 
                       bufs, mem_guard);
    Serialize<std::string>(&(st->state_->msg), bufs, mem_guard);
  } else {
    Serialize<int32_t>(mem_guard.AllocateElement<int32_t>(0),
                       bufs, mem_guard);    
  }
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::Status>(const char* buf, 
                                                    ps::Status* st, 
                                                    size_t* len,
                                                    MemGuard& mem_guard) {
  int32_t error_code;
  size_t field_len;
  Deserialize<int32_t>(buf, &error_code, &field_len, mem_guard);
  *len = field_len;
  if (error_code != 0) {
    std::string msg;
    Deserialize<std::string>(buf + *len, &msg, &field_len, mem_guard);
    *len += field_len;
    *st = ps::Status((ps::Status::ErrorCode)error_code, msg);
  } else {
    *st = ps::Status();
  }
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::DataType>(
    const ps::DataType* dt, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<int32_t>((int32_t*)dt, bufs, mem_guard);
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::DataType>(const char* buf, 
                                                      ps::DataType* dt, 
                                                      size_t* len,
                                                      MemGuard& mem_guard) {
  int32_t type;
  size_t field_len;
  Deserialize<int32_t>(buf, &type, &field_len, mem_guard);
  *dt = (ps::DataType)type;
  *len = sizeof(int32_t);
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::TensorShape>(
    const ps::TensorShape* ts, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(ts->Size()), 
                    bufs, mem_guard);
  Fragment size((char*)(&(ts->Dims()[0])), ts->Size() * sizeof(size_t));
  bufs->push_back(size);
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::TensorShape>(const char* buf, 
                                                         ps::TensorShape* ts, 
                                                         size_t* len,
                                                         MemGuard& mem_guard) {
  size_t field_len;
  size_t dim_len;
  SerializeHelper::Deserialize<size_t>(buf, &dim_len, &field_len, mem_guard);
  *len = field_len;
  std::vector<size_t> dims;
  dims.resize(dim_len);
  size_t data_len = dim_len * sizeof(size_t);
  memcpy(&(dims[0]), buf + *len, data_len);
  *len += data_len;
  *ts = TensorShape(dims);
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<std::unique_ptr<ps::Initializer> >(
    const std::unique_ptr<ps::Initializer>* data, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  size_t id;
  std::vector<Fragment> temp_bufs;
  SerializeAny<ps::Initializer>((ps::Initializer*)data->get(), 
                                &id, 
                                &temp_bufs, 
                                mem_guard);
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(id),
                    bufs, mem_guard);
  bufs->insert(bufs->end(), temp_bufs.begin(), temp_bufs.end());
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::Tensor>(
    const ps::Tensor* t, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  if (t->tensor_type_ != Tensor::TType::kContinuous) {
    return Status::ArgumentError("SegmentTensor can't be serialized");
  }
  size_t buffer_size = sizeof(ps::DataType) + (1 + t->state_->shape.Size()) * sizeof(size_t);
  char* buffer = mem_guard.AllocateBuffer(buffer_size);
  *(ps::DataType*)buffer = t->state_->type;
  *(size_t*)(buffer + sizeof(ps::DataType)) = t->state_->shape.Size();
  memcpy(buffer + sizeof(ps::DataType) + sizeof(size_t), &(t->state_->shape.dims_[0]), t->state_->shape.Size() * sizeof(size_t));
  bufs->push_back(Fragment{.base=buffer, .size=buffer_size});
  ps::Tensor::ContinuousState* state = dynamic_cast<ps::Tensor::ContinuousState*>(t->state_);
  size_t size = t->Shape().NumElements() * SizeOfType(t->Type());
  bufs->push_back(Fragment{.base=state->buffer, .size=size});
  Serialize<std::unique_ptr<ps::Initializer> >(&t->state_->initializer, 
      bufs, 
      mem_guard);
  return ps::Status::Ok();  
}

template <>  
ps::Status SerializeHelper::Serialize<std::vector<ps::Tensor> >(
    const std::vector<ps::Tensor>* tvec, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  if (tvec->size() == 1) {
    return SerializeHelper::SerializeVec<ps::Tensor>(tvec, bufs, mem_guard);
  }
  size_t buffer_size = sizeof(size_t);
  for (size_t i = 0; i < tvec->size(); i++) {
    const ps::Tensor* t = &tvec->at(i);
    buffer_size += sizeof(ps::DataType) + (1 + t->state_->shape.Size()) * sizeof(size_t);
    buffer_size += t->Shape().NumElements() * SizeOfType(t->Type());
  }
  char* buffer = mem_guard.AllocateBuffer(buffer_size);
  *(size_t*)buffer = tvec->size();
  size_t offset = sizeof(size_t);
  for (size_t i = 0; i < tvec->size(); i++) {
    const ps::Tensor* t = &tvec->at(i);
    if (t->tensor_type_ != Tensor::TType::kContinuous) {
      return Status::ArgumentError("SegmentTensor can't be serialized");
    }
    *(ps::DataType*)(buffer+offset) = t->state_->type;
    offset += sizeof(ps::DataType);
    *(size_t*)(buffer + offset) = t->state_->shape.Size();
    offset += sizeof(size_t);
    memcpy(buffer + offset, &(t->state_->shape.dims_[0]), t->state_->shape.Size() * sizeof(size_t));
    offset += t->state_->shape.Size() * sizeof(size_t);
    ps::Tensor::ContinuousState* state = dynamic_cast<ps::Tensor::ContinuousState*>(t->state_);
    size_t size = t->Shape().NumElements() * SizeOfType(t->Type());    
    memcpy(buffer + offset, state->buffer, size);
    offset += size;
  }
  bufs->push_back(Fragment{.base=buffer, .size=buffer_size});
  return Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::server::TensorSlices> (
    const ps::server::TensorSlices* data, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  const ps::server::TensorSlices& s = *data;
  const ps::Tensor* t = &s.tensor;
  size_t buffer_size = sizeof(ps::DataType) + sizeof(size_t);
  if (s.dim_part < 0) {
    buffer_size += s.tensor.Shape().Size() * sizeof(size_t);
  } else {
    buffer_size += (s.tensor.Shape().Size() - s.dim_part + 1) * sizeof(size_t);
  }
  buffer_size += s.slice_id.size() * s.slice_size * SizeOfType(t->Type());
  char* buffer = mem_guard.AllocateBuffer(buffer_size);
  *(ps::DataType*)(buffer) = t->state_->type;
  size_t offset = sizeof(ps::DataType);
  ps::TensorShape new_shape;
  if (s.dim_part < 0) {
    new_shape = t->state_->shape;
  } else {
    std::vector<size_t> dims(1, s.slice_id.size());
    if ((size_t)s.dim_part >= t->state_->shape.Size()) {
      return Status::ArgumentError("Slice dim_part Error");
    }
    dims.insert(dims.end(), t->state_->shape.Dims().begin() + s.dim_part, t->state_->shape.Dims().end());
    new_shape = TensorShape(dims);
  }
  *(size_t*)(buffer + offset) = new_shape.Size();
  offset += sizeof(size_t);
  memcpy(buffer + offset, &(new_shape.dims_[0]), new_shape.Size() * sizeof(size_t));
  offset += new_shape.Size() * sizeof(size_t);
  
  size_t chunk_size = s.slice_size * SizeOfType(t->Type());
  for (size_t j = 0; j < s.slice_id.size(); ++j) {
    if ((int64_t)s.slice_id[j] == ps::HashMap::NOT_ADD_ID) {
      memset(buffer + offset + j * chunk_size, 0, chunk_size);
    } else {
      memcpy(buffer + offset + j * chunk_size, t->Raw<void>(s.slice_id[j]), chunk_size);
    }
  }
  bufs->push_back(Fragment({.base=buffer, .size=buffer_size}));
  return ps::Status::Ok();  
}

template <>
ps::Status SerializeHelper::Deserialize<ps::Tensor>(const char* buf, 
                                                    ps::Tensor* t, 
                                                    size_t* len,
                                                    MemGuard& mem_guard) {
  size_t field_len;
  ps::DataType type;
  PS_CHECK_STATUS(Deserialize<ps::DataType>(buf, &type, &field_len, mem_guard));
  *len = field_len;
  ps::TensorShape shape({0});
  PS_CHECK_STATUS(Deserialize<ps::TensorShape>(buf + *len, &shape, &field_len, mem_guard));
  *len += field_len;
  const char* tensor_buffer = buf + *len;
  size_t buffer_len = 0;
  CASES(type, {
    buffer_len = shape.NumElements() * sizeof(T);
  });
  *len += buffer_len;
  size_t serialize_id = 0;
  PS_CHECK_STATUS(Deserialize<size_t>(buf + *len, &serialize_id, &field_len, mem_guard));
  *len += field_len;
  ps::Initializer* iz = nullptr;
  Fragment frag({.base=(char*)buf, .size=*len});
  //Allow no initializer(for Slices & TensorSlices)
  Status st = DeserializeAny<ps::Initializer>(serialize_id, &frag, *len, &iz, &field_len, mem_guard);
  if (st.IsOk()) {
    *len += field_len;
  } else {
    *len -= sizeof(size_t);
  }
  *t = Tensor(type, std::move(shape), const_cast<char*>(tensor_buffer), iz);
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<std::vector<ps::Tensor> >(const char* buf, 
    std::vector<ps::Tensor>* t, 
    size_t* len,
    MemGuard& mem_guard) {
  return SerializeHelper::DeserializeVec(buf, t, len, mem_guard);
}

// Serialzier for ps::Initializer
using Initializer = ps::Initializer;
using NoneInitializer = ps::initializer::NoneInitializer;
using ConstantInitializer = ps::initializer::ConstantInitializer;
using TruncatedNormalInitializer = ps::initializer::TruncatedNormalInitializer;
using NormalInitializer = ps::initializer::NormalInitializer;
using VarianceScalingInitializer = ps::initializer::VarianceScalingInitializer;
using UniformUnitScalingInitializer = ps::initializer::UniformUnitScalingInitializer;
using OrthogonalInitializer = ps::initializer::OrthogonalInitializer;

class NoneInitializerSerializer: 
    public Serializer<Initializer, NoneInitializer> {
 public:
  virtual ps::Status Serialize(NoneInitializer* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    return ps::Status::Ok();
  }
};

class NoneInitializerDeserializer: 
    public Deserializer<Initializer, NoneInitializer> {
 public:
  virtual ps::Status Deserialize(Fragment* buf, 
                                 size_t offset, 
                                 NoneInitializer** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    *len = 0;
    *result = new NoneInitializer();
    return ps::Status::Ok();
  }
};

class ConstantInitializerSerializer: 
    public Serializer<Initializer, ConstantInitializer> {
 public:
  virtual ps::Status Serialize(ConstantInitializer* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    return SerializeHelper::Serialize<double>(
        mem_guard.AllocateElement<double>(data->c_), 
        bufs, mem_guard);
  }
};

class ConstantInitializerDeserializer: 
    public Deserializer<Initializer, ConstantInitializer> {
 public:
  virtual ps::Status Deserialize(Fragment* buf, 
                                 size_t offset, 
                                 ConstantInitializer** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    size_t field_len;
    double value;
    SerializeHelper::Deserialize<double>(buf->base + offset, 
                                         &value,
                                         &field_len, 
                                         mem_guard);
    *len = field_len;
    *result = new ConstantInitializer(value);
    return ps::Status::Ok();
  }
};

class TruncatedNormalInitializerSerializer: 
    public Serializer<Initializer, TruncatedNormalInitializer> {
 public:
  virtual ps::Status Serialize(TruncatedNormalInitializer* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    SerializeHelper::Serialize<int>(&data->seed_, 
                                    bufs, 
                                    mem_guard);
    SerializeHelper::Serialize<float>(&data->mean_, 
                                      bufs, 
                                      mem_guard);
    SerializeHelper::Serialize<float>(&data->stddev_, 
                                      bufs, 
                                      mem_guard);
    return ps::Status::Ok();
  }
};

class TruncatedNormalInitializerDeserializer: 
    public Deserializer<Initializer, TruncatedNormalInitializer> {
 public:
  virtual ps::Status Deserialize(Fragment* buf, 
                                 size_t offset, 
                                 TruncatedNormalInitializer** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    size_t field_len;
    int seed;
    char* base = buf->base + offset;
    SerializeHelper::Deserialize<int>(base, 
                                      &seed, 
                                      &field_len, 
                                      mem_guard);
    *len = field_len;
    float mean;
    SerializeHelper::Deserialize<float>(base + *len, 
                                        &mean, 
                                        &field_len, 
                                        mem_guard);
    *len += field_len;
    float stddev;
    SerializeHelper::Deserialize<float>(base + *len, 
                                        &stddev, 
                                        &field_len, 
                                        mem_guard);
    *len += field_len;    
    *result = new TruncatedNormalInitializer(seed, mean, stddev);
    return ps::Status::Ok();
  }
};

class NormalInitializerSerializer: 
    public Serializer<Initializer, NormalInitializer> {
 public:
  virtual ps::Status Serialize(NormalInitializer* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    SerializeHelper::Serialize<int>(&data->seed_, 
                                    bufs, 
                                    mem_guard);
    SerializeHelper::Serialize<float>(&data->mean_, 
                                      bufs, 
                                      mem_guard);
    SerializeHelper::Serialize<float>(&data->stddev_, 
                                      bufs, 
                                      mem_guard);
    return ps::Status::Ok();
  }
};

class NormalInitializerDeserializer: 
    public Deserializer<Initializer, NormalInitializer> {
 public:
  virtual ps::Status Deserialize(Fragment* buf, 
                                 size_t offset, 
                                 NormalInitializer** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    size_t field_len;
    int seed;
    char* base = buf->base + offset;
    SerializeHelper::Deserialize<int>(base, 
                                      &seed, 
                                      &field_len, 
                                      mem_guard);
    *len = field_len;
    float mean;
    SerializeHelper::Deserialize<float>(base + *len, 
                                        &mean, 
                                        &field_len, 
                                        mem_guard);
    *len += field_len;
    float stddev;
    SerializeHelper::Deserialize<float>(base + *len, 
                                        &stddev, 
                                        &field_len, 
                                        mem_guard);
    *len += field_len;    
    *result = new NormalInitializer(seed, mean, stddev);
    return ps::Status::Ok();
  }
};

class OrthogonalInitializerSerializer: 
    public Serializer<Initializer, OrthogonalInitializer> {
 public:
  virtual ps::Status Serialize(OrthogonalInitializer* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    SerializeHelper::Serialize<int>(&data->seed_, 
                                    bufs, 
                                    mem_guard);
    SerializeHelper::Serialize<float>(&data->gain_, 
                                      bufs, 
                                      mem_guard);
    SerializeHelper::Serialize<int64_t>(&data->dim_, 
                                      bufs, 
                                      mem_guard);
    return ps::Status::Ok();
  }
};

class OrthogonalInitializerDeserializer: 
    public Deserializer<Initializer, OrthogonalInitializer> {
 public:
  virtual ps::Status Deserialize(Fragment* buf, 
                                 size_t offset, 
                                 OrthogonalInitializer** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    size_t field_len;
    int seed;
    char* base = buf->base + offset;
    SerializeHelper::Deserialize<int>(base, 
                                      &seed, 
                                      &field_len, 
                                      mem_guard);
    *len = field_len;
    float gain;
    SerializeHelper::Deserialize<float>(base + *len, 
                                        &gain, 
                                        &field_len, 
                                        mem_guard);
    *len += field_len;
    int64_t dim;
    SerializeHelper::Deserialize<int64_t>(base + *len, 
                                          &dim, 
                                          &field_len, 
                                          mem_guard);
    *len += field_len;    
    *result = new OrthogonalInitializer(dim, seed, gain);
    return ps::Status::Ok();
  }
};

class UniformUnitScalingInitializerSerializer: 
    public Serializer<Initializer, UniformUnitScalingInitializer> {
 public:
  virtual ps::Status Serialize(UniformUnitScalingInitializer* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    SerializeHelper::Serialize<int>(&data->seed_, 
                                    bufs, 
                                    mem_guard);
    SerializeHelper::Serialize<float>(&data->factor_, 
                                      bufs, 
                                      mem_guard);
    SerializeHelper::Serialize<ps::TensorShape>(&data->shape_, 
                                                bufs, 
                                                mem_guard);
    return ps::Status::Ok();
  }
};

class UniformUnitScalingInitializerDeserializer: 
    public Deserializer<Initializer, UniformUnitScalingInitializer> {
 public:
  virtual ps::Status Deserialize(Fragment* buf, 
                                 size_t offset, 
                                 UniformUnitScalingInitializer** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    size_t field_len;
    int seed;
    char* base = buf->base + offset;
    SerializeHelper::Deserialize<int>(base, 
                                      &seed, 
                                      &field_len, 
                                      mem_guard);
    *len = field_len;
    float factor;
    SerializeHelper::Deserialize<float>(base + *len, 
                                        &factor, 
                                        &field_len, 
                                        mem_guard);
    *len += field_len;
    ps::TensorShape shape;
    SerializeHelper::Deserialize<ps::TensorShape>(base + *len, 
                                                  &shape, 
                                                  &field_len, 
                                                  mem_guard);
    *len += field_len;    
    *result = new UniformUnitScalingInitializer(std::move(shape), 
                                                seed, 
                                                factor);
    return ps::Status::Ok();
  }
};

class VarianceScalingInitializerSerializer: 
    public Serializer<Initializer, VarianceScalingInitializer> {
 public:
  virtual ps::Status Serialize(VarianceScalingInitializer* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    SerializeHelper::Serialize<int>(&data->seed_, 
                                    bufs,
                                    mem_guard);
    SerializeHelper::Serialize<double>(&data->scale_, 
                                       bufs, 
                                       mem_guard);
    SerializeHelper::Serialize<std::string>(&data->mode_, 
                                            bufs, 
                                            mem_guard);
    SerializeHelper::Serialize<std::string>(&data->distribution_, 
                                            bufs, 
                                            mem_guard);
    SerializeHelper::Serialize<ps::TensorShape>(&data->full_shape_, 
                                                bufs, 
                                                mem_guard);
    return ps::Status::Ok();
  }
};

class VarianceScalingInitializerDeserializer: 
    public Deserializer<Initializer, VarianceScalingInitializer> {
 public:
  virtual ps::Status Deserialize(Fragment* buf, 
                                 size_t offset, 
                                 VarianceScalingInitializer** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    size_t field_len;
    int seed;
    char* base = buf->base + offset;
    SerializeHelper::Deserialize<int>(base, 
                                      &seed, 
                                      &field_len, 
                                      mem_guard);
    *len = field_len;
    double scale;
    SerializeHelper::Deserialize<double>(base + *len, 
                                         &scale, 
                                         &field_len, 
                                         mem_guard);
    *len += field_len;
    std::string mode;
    SerializeHelper::Deserialize<std::string>(base + *len, 
                                              &mode, 
                                              &field_len, 
                                              mem_guard);
    *len += field_len;
    std::string distribution;
    SerializeHelper::Deserialize<std::string>(base + *len, 
                                              &distribution, 
                                              &field_len, 
                                              mem_guard);
    *len += field_len;
    ps::TensorShape shape;
    SerializeHelper::Deserialize<ps::TensorShape>(base + *len, 
                                                  &shape, 
                                                  &field_len, 
                                                  mem_guard);
    *len += field_len;    
    *result = new VarianceScalingInitializer(std::move(shape), 
                                             seed, 
                                             scale,
                                             mode,
                                             distribution);
    return ps::Status::Ok();
  }
};

// Serializer for ps::WrapperData<T>
template <typename T>
class WrapperDataSerializer: public Serializer<ps::Data, ps::WrapperData<T>> {
 public:
  virtual ps::Status Serialize(ps::WrapperData<T>* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    return SerializeHelper::Serialize(&(data->Internal()), bufs, mem_guard);
  }
};

// Deserializer for ps::WrapperData<T>
template <typename T>
class WrapperDataDerializer: public Deserializer<ps::Data, ps::WrapperData<T> > {
 public:
  virtual ps::Status Deserialize(Fragment* buf, 
                                 size_t offset, 
                                 ps::WrapperData<T>** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    char* buf_start = reinterpret_cast<char*>(buf->base + offset);
    *result = new ps::WrapperData<T>();
    return SerializeHelper::Deserialize(buf_start, 
                                        &(*result)->Internal(), 
                                        len, 
                                        mem_guard);
  }
};

template <>
class WrapperDataDerializer<ps::TensorShape>: 
    public Deserializer<ps::Data, ps::WrapperData<ps::TensorShape> > {
 public:
  virtual ps::Status Deserialize(Fragment* frag, 
                                 size_t offset, 
                                 ps::WrapperData<ps::TensorShape>** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    char* buf = reinterpret_cast<char*>(frag->base + offset);
    size_t field_len;
    size_t dim_len;
    SerializeHelper::Deserialize<size_t>(buf, 
                                         &dim_len, 
                                         &field_len, 
                                         mem_guard);
    *len = field_len;
    std::vector<size_t> dims;
    dims.resize(dim_len);
    size_t data_len = dim_len * sizeof(size_t);
    memcpy(&(dims[0]), buf + *len, data_len);
    *len += data_len;
    *result = new ps::WrapperData<ps::TensorShape>(TensorShape(std::move(dims)));
    return ps::Status::Ok();
  }
};

template <>
class WrapperDataSerializer<ps::server::TensorSlices>: 
    public Serializer<ps::Data, ps::WrapperData<ps::server::TensorSlices>, ps::WrapperData<ps::Tensor> > {
 public:
  virtual ps::Status Serialize(ps::WrapperData<ps::server::TensorSlices>* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    return SerializeHelper::Serialize<ps::server::TensorSlices>(&data->Internal(), bufs, mem_guard);
  }
};

template <>
class WrapperDataSerializer<std::vector<ps::server::TensorSlices> >:
    public Serializer<ps::Data, ps::WrapperData<std::vector<ps::server::TensorSlices> >, ps::WrapperData<std::vector<ps::Tensor> > > {
public:
  virtual ps::Status Serialize(ps::WrapperData<std::vector<ps::server::TensorSlices> >* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    size_t buffer_size = sizeof(size_t);
    for (size_t i = 0; i < data->Internal().size(); i++) {
      buffer_size += sizeof(ps::DataType) + sizeof(size_t);
      const ps::server::TensorSlices& s = data->Internal()[i];
      if (s.dim_part < 0) {
        buffer_size += s.tensor.Shape().Size() * sizeof(size_t);
      } else {
        buffer_size += (s.tensor.Shape().Size() - s.dim_part + 1) * sizeof(size_t);
      }
      buffer_size += s.slice_id.size() * s.slice_size * SizeOfType(s.tensor.Type());
    }
    char* buffer = mem_guard.AllocateBuffer(buffer_size);
    *((size_t*)buffer) = data->Internal().size();
    size_t offset = sizeof(size_t);
    for (size_t i = 0; i < data->Internal().size(); i++) {
      const ps::server::TensorSlices& s = data->Internal()[i];
      const ps::Tensor* t = &s.tensor;
      *(ps::DataType*)(buffer + offset) = t->state_->type;
      offset += sizeof(ps::DataType);
      ps::TensorShape new_shape;
      if (s.dim_part < 0) {
        new_shape = t->state_->shape;
      } else {
        std::vector<size_t> dims(1, s.slice_id.size());
        if ((size_t)s.dim_part >= t->state_->shape.Size()) {
          return Status::ArgumentError("Slice dim_part Error");
        }
        dims.insert(dims.end(), t->state_->shape.Dims().begin() + s.dim_part, t->state_->shape.Dims().end());
        new_shape = TensorShape(dims);
      }
      *(size_t*)(buffer + offset) = new_shape.Size();
      offset += sizeof(size_t);
      memcpy(buffer + offset, &(new_shape.dims_[0]), new_shape.Size() * sizeof(size_t));
      offset += new_shape.Size() * sizeof(size_t);

      size_t chunk_size = s.slice_size * SizeOfType(t->Type());
      for (size_t j = 0; j < s.slice_id.size(); ++j) {
        if ((int64_t)s.slice_id[j] == ps::HashMap::NOT_ADD_ID) {
          memset(buffer + offset + j * chunk_size, 0, chunk_size);
        } else {
          memcpy(buffer + offset + j * chunk_size, t->Raw<void>(s.slice_id[j]), chunk_size);
        }
      }
      offset += s.slice_id.size() * chunk_size;
    }
    Fragment buf(buffer, buffer_size);
    bufs->push_back(std::move(buf));
    return ps::Status::Ok();
  }
};

template <>
class WrapperDataSerializer<std::vector<ps::server::Slices> >:
    public Serializer<ps::Data, ps::WrapperData<std::vector<ps::server::Slices> >, ps::WrapperData<std::vector<ps::Tensor> > > {
public:
  virtual ps::Status Serialize(ps::WrapperData<std::vector<ps::server::Slices> >* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    size_t buffer_size = sizeof(size_t);
    for (size_t i = 0; i < data->Internal().size(); i++) {
      buffer_size += sizeof(ps::DataType) + sizeof(size_t);
      const ps::server::Slices& s = data->Internal()[i];
      ps::Tensor* tensor = s.variable->GetData();
      if (s.dim_part < 0) {
        buffer_size += tensor->Shape().Size() * sizeof(size_t);
      } else {
        buffer_size += (tensor->Shape().Size() - s.dim_part + 1) * sizeof(size_t);
      }
      buffer_size += s.slice_id.size() * s.slice_size * SizeOfType(tensor->Type());
    }
    char* buffer = mem_guard.AllocateBuffer(buffer_size);
    *((size_t*)buffer) = data->Internal().size();
    size_t offset = sizeof(size_t);
    for (size_t i = 0; i < data->Internal().size(); i++) {
      const ps::server::Slices& s = data->Internal()[i];
      const ps::Tensor* t = s.variable->GetData();
      *(ps::DataType*)(buffer + offset) = t->state_->type;
      offset += sizeof(ps::DataType);
      ps::TensorShape new_shape;
      if (s.dim_part < 0) {
        new_shape = t->state_->shape;
      } else {
        std::vector<size_t> dims(1, s.slice_id.size());
        if ((size_t)s.dim_part >= t->state_->shape.Size()) {
          return Status::ArgumentError("Slice dim_part Error");
        }
        dims.insert(dims.end(), t->state_->shape.Dims().begin() + s.dim_part, t->state_->shape.Dims().end());
        new_shape = TensorShape(dims);
      }
      *(size_t*)(buffer + offset) = new_shape.Size();
      offset += sizeof(size_t);
      memcpy(buffer + offset, &(new_shape.dims_[0]), new_shape.Size() * sizeof(size_t));
      offset += new_shape.Size() * sizeof(size_t);

      size_t chunk_size = s.slice_size * SizeOfType(t->Type());
      for (size_t j = 0; j < s.slice_id.size(); ++j) {
        if ((int64_t)s.slice_id[j] == ps::HashMap::NOT_ADD_ID) {
          memset(buffer + offset + j * chunk_size, 0, chunk_size);
        } else {
          memcpy(buffer + offset + j * chunk_size, t->Raw<void>(s.slice_id[j]), chunk_size);
        }
      }
      offset += s.slice_id.size() * chunk_size;
    }
    Fragment buf(buffer, buffer_size);
    bufs->push_back(std::move(buf));
    return ps::Status::Ok();
  }
};

template <>
class WrapperDataDerializer<std::unique_ptr<Initializer> >: 
    public Deserializer<ps::Data, ps::WrapperData<std::unique_ptr<Initializer> > > {
 public:
  virtual ps::Status Deserialize(Fragment* frag, 
                                 size_t offset, 
                                 ps::WrapperData<std::unique_ptr<Initializer> >** result, 
                                 size_t* len,
                                 MemGuard& mem_guard) {
    size_t field_len;
    size_t id;
    SerializeHelper::Deserialize<size_t>(frag->base + offset, 
                                         &id, 
                                         &field_len, 
                                         mem_guard);
    *len = field_len;
    ps::Initializer* data = nullptr;
    ps::serializer::DeserializeAny<ps::Initializer>(
        id, frag, *len + offset, &data, &field_len, mem_guard);
    *len += field_len;
    *result = new ps::WrapperData<std::unique_ptr<Initializer> >(data);
    return ps::Status::Ok();
  }
};

} // namespace serializer
} // namespace ps

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<int8_t>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<int8_t>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<int16_t>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<int16_t>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<int32_t>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<int32_t>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<int64_t>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<int64_t>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<uint8_t>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<uint8_t>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<uint16_t>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<uint16_t>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<uint32_t>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<uint32_t>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<uint64_t>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<uint64_t>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<float>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<float>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<double>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<double>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<bool>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<bool>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<double> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::vector<double> >);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<float> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::vector<float> >);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<bool> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::vector<bool> >);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<int64_t> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::vector<int64_t> >);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<int> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::vector<int> >);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::string>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::string>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<std::string> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::vector<std::string> >);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::Status>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::Status>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::DataType>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::DataType>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::TensorShape>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::TensorShape>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::Tensor>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::Tensor>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<ps::Tensor> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::vector<ps::Tensor> >);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::unique_ptr<ps::Initializer> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::unique_ptr<ps::Initializer> >);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::server::TensorSlices>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<ps::server::TensorSlices> >);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::vector<ps::server::Slices> >);

SERIALIZER_REGISTER(ps::serializer::NoneInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::NoneInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::ConstantInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::ConstantInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::TruncatedNormalInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::TruncatedNormalInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::NormalInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::NormalInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::OrthogonalInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::OrthogonalInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::UniformUnitScalingInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::UniformUnitScalingInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::VarianceScalingInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::VarianceScalingInitializerDeserializer);

#endif // PS_COMMON_SERIALIZE_HELPER_H_

