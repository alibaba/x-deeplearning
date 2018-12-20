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
    T* buf = const_cast<T*>(data);
    Fragment frag(reinterpret_cast<char*>(buf), sizeof(T));
    bufs->push_back(frag);
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
  Serialize<ps::DataType>(&t->state_->type, bufs, mem_guard);
  Serialize<ps::TensorShape>(&t->state_->shape, bufs, mem_guard);
  size_t size = 0;
  CASES(t->Type(), {
    size = t->Shape().NumElements() * sizeof(T);
  });
  Fragment frag(t->state_->buffer, size);
  bufs->push_back(frag);
  Serialize<std::unique_ptr<ps::Initializer> >(&t->state_->initializer, 
                                               bufs, 
                                               mem_guard);
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
  DeserializeAny<ps::Initializer>(serialize_id, &frag, *len, &iz, &field_len, mem_guard);
  *len += field_len;
  *t = Tensor(type, std::move(shape), const_cast<char*>(tensor_buffer), iz);
  return ps::Status::Ok();
}

// Serialzier for ps::Initializer
using Initializer = ps::Initializer;
using NoneInitializer = ps::initializer::NoneInitializer;
using ConstantInitializer = ps::initializer::ConstantInitializer;
using TruncatedNormalInitializer = ps::initializer::TruncatedNormalInitializer;
using NormalInitializer = ps::initializer::NormalInitializer;
using VarianceScalingInitializer = ps::initializer::VarianceScalingInitializer;
using UniformUnitScalingInitializer = ps::initializer::UniformUnitScalingInitializer;

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
class WrapperDataSerializer<ps::server::Slices>: 
    public Serializer<ps::Data, ps::WrapperData<ps::server::Slices>, ps::WrapperData<ps::Tensor> > {
 public:
  virtual ps::Status Serialize(ps::WrapperData<ps::server::Slices>* data, 
                               std::vector<Fragment>* bufs,
                               MemGuard& mem_guard) {
    ps::server::Slices& s = data->Internal();
    ps::Tensor* t = s.variable->data_.get();
    SerializeHelper::Serialize<ps::DataType>(
        &t->state_->type, bufs, mem_guard);
    ps::TensorShape new_shape;
    if (s.dim_part < 0) {
      new_shape = t->state_->shape;
    } else {
      std::vector<size_t> dims(1, s.slice_size);
      if ((size_t)s.dim_part > t->state_->shape.Size()) {
        return Status::ArgumentError("Slice dim_part Error");
      }
      dims.insert(dims.end(), t->state_->shape.Dims().begin() + s.dim_part, t->state_->shape.Dims().end());
      new_shape = TensorShape(dims);
      new_shape.Set(0, s.slice_id.size());
    }
    SerializeHelper::Serialize<size_t>(
        mem_guard.AllocateElement<size_t>(new_shape.Size()),
        bufs, mem_guard);
    size_t buf_size = t->Shape().Size() * sizeof(size_t);
    char* shape_buf = mem_guard.AllocateBuffer(buf_size);
    memcpy(shape_buf, &(new_shape.dims_[0]), buf_size);
    bufs->push_back(Fragment({.base=shape_buf, .size=buf_size}));
    char* base = t->Raw<char>();
    for (size_t i = 0; i < s.slice_id.size(); ++i) {
      CASES(t->Type(), {
        bufs->push_back(Fragment({.base=base + s.slice_id[i] * s.slice_size * sizeof(T), 
            .size=sizeof(T) * s.slice_size}));
      });
    }

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
    static char zero_buffer[1<<16] = {0};
    ps::server::TensorSlices& s = data->Internal();
    ps::Tensor* t = &s.tensor;
    SerializeHelper::Serialize<ps::DataType>(
        &t->state_->type, bufs, mem_guard);
    ps::TensorShape new_shape;
    if (s.dim_part < 0) {
      new_shape = t->state_->shape;
    } else {
      std::vector<size_t> dims(1, s.slice_size);
      if ((size_t)s.dim_part > t->state_->shape.Size()) {
        return Status::ArgumentError("Slice dim_part Error");
      }
      dims.insert(dims.end(), t->state_->shape.Dims().begin() + s.dim_part, t->state_->shape.Dims().end());
      new_shape = TensorShape(dims);
      new_shape.Set(0, s.slice_id.size());
    }
    SerializeHelper::Serialize<size_t>(
        mem_guard.AllocateElement<size_t>(new_shape.Size()),
        bufs, mem_guard);
    size_t buf_size = t->Shape().Size() * sizeof(size_t);
    char* shape_buf = mem_guard.AllocateBuffer(buf_size);
    memcpy(shape_buf, &(new_shape.dims_[0]), buf_size);
    bufs->push_back(Fragment({.base=shape_buf, .size=buf_size}));

    if (s.slice_id.size() <= 16) {
      char* base = t->Raw<char>();
      CASES(t->Type(), {
	  for (size_t i = 0; i < s.slice_id.size(); ++i) {
          if ((int64_t)s.slice_id[i] == ps::HashMap::NOT_ADD_ID) {
          bufs->push_back(Fragment({.base=zero_buffer, .size=sizeof(T) * s.slice_size}));
        } else {
          bufs->push_back(Fragment({.base=base + s.slice_id[i] * s.slice_size * sizeof(T), 
		    .size=sizeof(T) * s.slice_size}));
        }
	  }
	});
    } else {
      size_t buf_size = 0;
      CASES(t->Type(), {
	  buf_size = s.slice_id.size() * s.slice_size * sizeof(T);
	});
      char* slice_buf = mem_guard.AllocateBuffer(buf_size);
      char* base = t->Raw<char>();
      CASES(t->Type(), {
	  size_t chunk_size = s.slice_size * sizeof(T);
	  for (size_t i = 0; i < s.slice_id.size(); ++i) {
        if ((int64_t)s.slice_id[i] == ps::HashMap::NOT_ADD_ID) {
          memset(slice_buf + i * chunk_size, 0, chunk_size);
        } else {
          memcpy(slice_buf + i * chunk_size, base + s.slice_id[i] * chunk_size, chunk_size);
        }
      }
	});
      bufs->push_back(Fragment({.base=slice_buf, .size=buf_size}));
    }

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

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::string>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::string>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::Status>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::Status>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::DataType>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::DataType>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::TensorShape>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::TensorShape>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::Tensor>);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<ps::Tensor>);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<std::unique_ptr<ps::Initializer> >);
DESERIALIZER_REGISTER(ps::serializer::WrapperDataDerializer<std::unique_ptr<ps::Initializer> >);

SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::server::Slices>);
SERIALIZER_REGISTER(ps::serializer::WrapperDataSerializer<ps::server::TensorSlices>);

SERIALIZER_REGISTER(ps::serializer::NoneInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::NoneInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::ConstantInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::ConstantInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::TruncatedNormalInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::TruncatedNormalInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::NormalInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::NormalInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::UniformUnitScalingInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::UniformUnitScalingInitializerDeserializer);

SERIALIZER_REGISTER(ps::serializer::VarianceScalingInitializerSerializer);
DESERIALIZER_REGISTER(ps::serializer::VarianceScalingInitializerDeserializer);

#endif // PS_COMMON_SERIALIZE_HELPER_H_

