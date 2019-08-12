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

#ifndef PS_MESSAGE_MESSAGE_SERIALIZER_H_
#define PS_MESSAGE_MESSAGE_SERIALIZER_H_

#include "ps-plus/common/serialize_helper.h"

#include "server_info.h"
#include "cluster_info.h"
#include "variable_info.h"
#include "udf_chain_register.h"
#include "streaming_model_infos.h"
#include "worker_state.h"

namespace ps {
namespace serializer {

template <>  
ps::Status SerializeHelper::Serialize<ps::ServerInfo>(
    const ps::ServerInfo* si, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<ps::ServerType>(&(si->server_type_), bufs, mem_guard);
  Serialize<ps::ServerId>(&(si->id_), bufs, mem_guard);
  Serialize<ps::Version>(&(si->version_), bufs, mem_guard);
  Serialize<std::string>(&(si->ip_), bufs, mem_guard);
  Serialize<uint16_t>(&(si->port_), bufs, mem_guard);
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::ServerInfo>(
    const char* buf, 
    ps::ServerInfo* si, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t field_len;
  Deserialize<ps::ServerType>(buf, &(si->server_type_), &field_len, mem_guard);
  *len = field_len;
  Deserialize<ps::ServerId>(buf + *len, &(si->id_), &field_len, mem_guard);
  *len += field_len;
  Deserialize<ps::Version>(buf + *len, &(si->version_), &field_len, mem_guard);
  *len += field_len;
  Deserialize<std::string>(buf + *len, &(si->ip_), &field_len, mem_guard);
  *len += field_len;
  Deserialize<uint16_t>(buf + *len, &(si->port_), &field_len, mem_guard);
  *len += field_len;
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::ClusterInfo>(
    const ps::ClusterInfo* ci, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(ci->servers_.size()), 
                    bufs, mem_guard);
  for (auto& it: ci->servers_) {
    Serialize<ps::ServerInfo>(&it, bufs, mem_guard);
  }
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(ci->server_size_.size()), 
                    bufs, mem_guard);
  for (auto it: ci->server_size_) {
    Serialize<size_t>(mem_guard.AllocateElement<size_t>(it), bufs, mem_guard);
  }

  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::ClusterInfo>(
    const char* buf, 
    ps::ClusterInfo* ci, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t size;
  size_t field_len;
  Deserialize<size_t>(buf, &size, &field_len, mem_guard);
  *len = field_len;
  for (size_t i = 0; i < size; ++i) {
    ps::ServerInfo si;
    Deserialize<ps::ServerInfo>(buf + *len, &si, &field_len, mem_guard);
    ci->servers_.push_back(si);
    *len += field_len;
  }
  Deserialize<size_t>(buf + *len, &size, &field_len, mem_guard);
  *len += field_len;
  for (size_t i = 0; i < size; ++i) {
    size_t s;
    Deserialize<size_t>(buf + *len, &s, &field_len, mem_guard);
    ci->server_size_.push_back(s);
    *len += field_len;
  }

  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::VariableInfo>(
    const ps::VariableInfo* vi, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<int32_t>(reinterpret_cast<const int32_t*>(&(vi->type)), 
                     bufs, mem_guard);
  Serialize<std::string>(&(vi->name), bufs, mem_guard);
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(vi->parts.size()), 
                    bufs, mem_guard);
  for (auto& it: vi->parts) {
    Serialize<size_t>(&it.server, bufs, mem_guard);
    Serialize<size_t>(&it.size, bufs, mem_guard);
  }

  Serialize<size_t>(mem_guard.AllocateElement<size_t>(vi->shape.size()), 
                    bufs, mem_guard);
  for (size_t i = 0; i < vi->shape.size(); ++i) {
    Serialize<int64_t>(&(vi->shape[i]), bufs, mem_guard);
  }

  Serialize<int32_t>(reinterpret_cast<const int32_t*>(&(vi->datatype)), 
                     bufs, mem_guard);
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(vi->args.size()), 
                    bufs, mem_guard);

  for (auto& it: vi->args) {
    Serialize<std::string>(&it.first, bufs, mem_guard);
    Serialize<std::string>(&it.second, bufs, mem_guard);
  }

  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::VariableInfo>(
    const char* buf, 
    ps::VariableInfo* vi, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t field_len;
  int32_t type;
  Deserialize<int32_t>(buf, &type, &field_len, mem_guard);  
  vi->type = (ps::VariableInfo::Type)type;
  *len = field_len;
  Deserialize<std::string>(buf + *len, &(vi->name), &field_len, mem_guard);    
  *len += field_len;
  size_t part_size;
  Deserialize<size_t>(buf + *len, &part_size, &field_len, mem_guard);  
  *len += field_len;
  vi->parts.resize(part_size);
  for (size_t i = 0; i < part_size; ++i) {
    Deserialize<size_t>(buf + *len, &(vi->parts[i].server), &field_len, mem_guard);  
    *len += field_len;    
    Deserialize<size_t>(buf + *len, &(vi->parts[i].size), &field_len, mem_guard);  
    *len += field_len;    
  }

  size_t shape_size;
  Deserialize<size_t>(buf + *len, &shape_size, &field_len, mem_guard);  
  *len += field_len;    
  vi->shape.resize(shape_size);
  for (size_t i = 0; i < shape_size; ++i) {
    Deserialize<int64_t>(buf + *len, &(vi->shape[i]), &field_len, mem_guard);      
    *len += field_len;    
  }

  int32_t data_type;
  Deserialize<int32_t>(buf + *len, &data_type, &field_len, mem_guard);  
  vi->datatype = (ps::types::DataType)data_type;
  *len += field_len;  

  size_t arg_size;
  Deserialize<size_t>(buf + *len, &arg_size, &field_len, mem_guard);    
  *len += field_len;
  for (size_t i = 0; i < arg_size; ++i) {
    std::string key;
    Deserialize<std::string>(buf + *len, &key, &field_len, mem_guard);        
    *len += field_len;
    std::string value;
    Deserialize<std::string>(buf + *len, &value, &field_len, mem_guard);        
    *len += field_len;
    vi->args.emplace(key, value);
  }

  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::VariableInfoCollection>(
    const ps::VariableInfoCollection* vic, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(vic->infos.size()), 
                    bufs, mem_guard);
  for (size_t i = 0; i < vic->infos.size(); ++i) {
    Serialize<ps::VariableInfo>(&(vic->infos[i]), bufs, mem_guard);
  }

  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::VariableInfoCollection>(
    const char* buf, 
    ps::VariableInfoCollection* vic, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t size;
  size_t field_len;
  Deserialize<size_t>(buf, &size, &field_len, mem_guard);  
  *len = field_len;
  vic->infos.resize(size);
  for (size_t i = 0; i < size; ++i) {
    Deserialize<ps::VariableInfo>(buf + *len, 
                                  &(vic->infos[i]), 
                                  &field_len, 
                                  mem_guard);
    *len += field_len;
  }

  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::UdfChainRegister::UdfDef>(
    const ps::UdfChainRegister::UdfDef* udf, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(udf->inputs.size()), 
                    bufs, mem_guard);  
  for (size_t i = 0; i < udf->inputs.size(); ++i) {
    Serialize<int>(&(udf->inputs[i].first), bufs, mem_guard);
    Serialize<int>(&(udf->inputs[i].second), bufs, mem_guard);
  }

  Serialize<std::string>(&(udf->udf_name), bufs, mem_guard);  
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::UdfChainRegister::UdfDef>(
    const char* buf, 
    ps::UdfChainRegister::UdfDef* udf, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t size;
  size_t field_len;
  Deserialize<size_t>(buf, &size, &field_len, mem_guard);  
  *len = field_len;
  udf->inputs.resize(size);
  for (size_t i = 0; i < size; ++i) {
    Deserialize<int>(buf + *len, &(udf->inputs[i].first), &field_len, mem_guard);      
    *len += field_len;
    Deserialize<int>(buf + *len, &(udf->inputs[i].second), &field_len, mem_guard);      
    *len += field_len;
  }

  Deserialize<std::string>(buf + *len, &(udf->udf_name), &field_len, mem_guard);    
  *len += field_len;
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::UdfChainRegister>(
    const ps::UdfChainRegister* udf_chain, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<size_t>(&(udf_chain->hash), bufs, mem_guard);
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(udf_chain->udfs.size()), 
                    bufs, mem_guard);  
  for (size_t i = 0; i < udf_chain->udfs.size(); ++i) {
    Serialize<ps::UdfChainRegister::UdfDef>(&(udf_chain->udfs[i]), 
                                            bufs, 
                                            mem_guard);
  }

  Serialize<size_t>(mem_guard.AllocateElement<size_t>(udf_chain->outputs.size()), 
                    bufs, mem_guard);    
  for (size_t i = 0; i < udf_chain->outputs.size(); ++i) {
    Serialize<int>(&(udf_chain->outputs[i].first), bufs, mem_guard);    
    Serialize<int>(&(udf_chain->outputs[i].second), bufs, mem_guard);    
  }
  
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::UdfChainRegister>(
    const char* buf, 
    ps::UdfChainRegister* udf_chain, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t field_len;
  Deserialize<size_t>(buf, &(udf_chain->hash), &field_len, mem_guard);  
  *len = field_len;
  size_t udf_size;
  Deserialize<size_t>(buf + *len, &udf_size, &field_len, mem_guard);    
  *len += field_len;
  udf_chain->udfs.resize(udf_size);
  for (size_t i = 0; i < udf_size; ++i) {
    Deserialize<ps::UdfChainRegister::UdfDef>(
        buf + *len, 
        &(udf_chain->udfs[i]), 
        &field_len, 
        mem_guard);    
    *len += field_len;
  }

  size_t outputs_size;
  Deserialize<size_t>(buf + *len, &outputs_size, &field_len, mem_guard);      
  *len += field_len;
  udf_chain->outputs.resize(outputs_size);
  for (size_t i = 0; i < outputs_size; ++i) {
    Deserialize<int>(buf + *len, 
                     &(udf_chain->outputs[i].first), 
                     &field_len, 
                     mem_guard);
    *len += field_len;
    Deserialize<int>(buf + *len, 
                     &(udf_chain->outputs[i].second), 
                     &field_len, 
                     mem_guard);
    *len += field_len;
  }

  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::DenseVarNames>(
    const ps::DenseVarNames* name, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(name->names.size()), bufs, mem_guard);
  for (size_t i = 0; i < name->names.size(); i++) {
    Serialize<std::string>(&(name->names[i]), bufs, mem_guard);
  }
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::DenseVarNames>(
    const char* buf, 
    ps::DenseVarNames* name, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t size;
  size_t field_len;
  Deserialize<size_t>(buf, &(size), &field_len, mem_guard);
  *len = field_len;
  name->names.resize(size);
  for (size_t i = 0; i < size; i++) {
    Deserialize<std::string>(buf + *len, &(name->names[i]), &field_len, mem_guard);
    *len += field_len;
  }
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::DenseVarValues>(
    const ps::DenseVarValues* value, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<size_t>(mem_guard.AllocateElement<size_t>(value->values.size()), bufs, mem_guard);
  for (size_t i = 0; i < value->values.size(); i++) {
    Serialize<std::string>(&(value->values[i].name), bufs, mem_guard);
    Serialize<size_t>(&(value->values[i].offset), bufs, mem_guard);
    Serialize<ps::Tensor>(&(value->values[i].data), bufs, mem_guard);
  }
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::DenseVarValues>(
    const char* buf, 
    ps::DenseVarValues* value, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t size;
  size_t field_len;
  Deserialize<size_t>(buf, &(size), &field_len, mem_guard);
  *len = field_len;
  value->values.resize(size);
  for (size_t i = 0; i < size; i++) {
    Deserialize<std::string>(buf + *len, &(value->values[i].name), &field_len, mem_guard);
    *len += field_len;
    Deserialize<size_t>(buf + *len, &(value->values[i].offset), &field_len, mem_guard);
    *len += field_len;
    Deserialize<ps::Tensor>(buf + *len, &(value->values[i].data), &field_len, mem_guard);
    *len += field_len;
  }
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<ps::WorkerState>(
    const ps::WorkerState* ws, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  Serialize<size_t>(&(ws->begin_), bufs, mem_guard);
  Serialize<size_t>(&(ws->end_), bufs, mem_guard);
  Serialize<size_t>(&(ws->epoch_), bufs, mem_guard);
  Serialize<std::string>(&(ws->path_), bufs, mem_guard);
  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<ps::WorkerState>(
    const char* buf, 
    ps::WorkerState* ws, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t field_len;
  Deserialize<size_t>(buf, &(ws->begin_), &field_len, mem_guard);
  *len = field_len;
  Deserialize<size_t>(buf + *len, &(ws->end_), &field_len, mem_guard);
  *len += field_len;
  Deserialize<size_t>(buf + *len, &(ws->epoch_), &field_len, mem_guard);
  *len += field_len;
  Deserialize<std::string>(buf + *len, &(ws->path_), &field_len, mem_guard);
  *len += field_len;
  return ps::Status::Ok();
}

template <>  
ps::Status SerializeHelper::Serialize<std::vector<ps::WorkerState> >(
    const std::vector<ps::WorkerState>* vws, 
    std::vector<Fragment>* bufs,
    MemGuard& mem_guard) {
  size_t vec_len = vws->size();
  Serialize<size_t>(&vec_len, bufs, mem_guard);  
  for (auto& ws: *vws) {
    Serialize<size_t>(&(ws.begin_), bufs, mem_guard);
    Serialize<size_t>(&(ws.end_), bufs, mem_guard);
    Serialize<size_t>(&(ws.epoch_), bufs, mem_guard);
    Serialize<std::string>(&(ws.path_), bufs, mem_guard);
  }

  return ps::Status::Ok();
}

template <>
ps::Status SerializeHelper::Deserialize<std::vector<ps::WorkerState> >(
    const char* buf, 
    std::vector<ps::WorkerState>* vws, 
    size_t* len,
    MemGuard& mem_guard) {
  size_t vec_len = 0;
  size_t field_len;
  Deserialize<size_t>(buf, &(vec_len), &field_len, mem_guard);
  *len = field_len;
  for (size_t i = 0; i < vec_len; ++i) {
    ps::WorkerState ws;
    Deserialize<size_t>(buf, &(ws.begin_), &field_len, mem_guard);
    *len += field_len;
    Deserialize<size_t>(buf + *len, &(ws.end_), &field_len, mem_guard);
    *len += field_len;
    Deserialize<size_t>(buf + *len, &(ws.epoch_), &field_len, mem_guard);
    *len += field_len;
    Deserialize<std::string>(buf + *len, &(ws.path_), &field_len, mem_guard);
    *len += field_len;
    vws->push_back(ws);
  }

  return ps::Status::Ok();
}

} // namespace serializer
} // namespace ps

#endif // PS_MESSAGE_MESSAGE_SERIALIZER_H_

