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

#include <chrono>
#include "ps-plus/server/checkpoint_utils.h"
#include "ps-plus/common/serializer.h"
#include "ps-plus/common/logging.h"
#include <map>
#define CK_CHECK_STATUS(STATUS, STATUS_RET, COUNTER, OK) do { Status st = STATUS_RET; if (!st.IsOk()) {STATUS = st; if (--COUNTER == 0) {OK.set_value(true);} return;}} while(0);

namespace ps {
namespace server {

CheckpointUtils::CheckpointUtils(const VariableInfoCollection& infos) : infos_(infos) {
}

Status CheckpointUtils::LoadVariables(
    const VariableInfoCollection& infos,
    size_t id,
    std::unordered_map<std::string, std::unique_ptr<Variable>>* vars) {
  std::atomic<size_t> counter(0);
  std::map<std::string, VariableInfo> source_infos;
  for (auto&& item : infos_.infos) {
    auto iter = item.args.find(VariableInfo::ORIGIN_NAME);
    if (iter != item.args.end()) {
      source_infos[iter->second] = item;
    } else {
      source_infos[item.name] = item;   
    }
  }

  for (auto&& to : infos.infos) {
    auto n = to.args.find(VariableInfo::ORIGIN_NAME); 
    std::string name = n == to.args.end() ? to.name : n->second;    
    auto iter = source_infos.find(name);
    if (iter != source_infos.end()) {
      for (size_t i = 0; i < to.parts.size(); i++) {
        auto part = to.parts[i];
        if (part.server == id) {
          (*vars)[to.name] = std::unique_ptr<Variable>(nullptr);
          ++counter;
          break;
        }
      }
    }
  }
  if (counter == 0) {
    return Status::Ok();
  }
  PS_CHECK_STATUS(MultiThreadDo(infos.infos.size(), [&](const Range& range) {
        for (size_t si = range.begin; si < range.end; ++si) {
          auto& to = infos.infos[si];
          auto n = to.args.find(VariableInfo::ORIGIN_NAME);
          std::string name = n == to.args.end() ? to.name : n->second;
          auto iter = source_infos.find(name);
          if (iter == source_infos.end()) {
            continue;
          }
          size_t beg = 0;
          for (size_t i = 0; i < to.parts.size(); i++) {
            auto part = to.parts[i];
            size_t end = beg + part.size;
            if (part.server == id) {
              LOG(INFO) << "Loading variable " << name << " part " << i;
              VariableInfo vi = iter->second;
              auto p = to.args.find(VariableInfo::ORIGIN_FILE_PATH);
              if (p != to.args.end()) {
                vi.args[VariableInfo::ORIGIN_FILE_PATH] = p->second;
              }
              Status st = MergeLoadVariable(to.name, vi, beg, end, &(*vars)[to.name]);
              if (!st.IsOk()) {
                if ((st.Code() == Status::ErrorCode::kUnknown || st.Code() == Status::ErrorCode::kNotFound) && vi.args.find("save") != vi.args.end() && vi.args["save"] == "false") {
                  LOG(INFO) << "Variable[" << to.name << "] not load.";
                  vars->erase(to.name);
                  continue;
                }
                LOG(WARNING) << "Variable[" << to.name <<  "] MergeLoadVariable failed, status[" << st.Msg() << "].";
                vars->erase(to.name);
                return st;
              } else {
                LOG(INFO) << "Variable[" << to.name << "] MergeLoadVariable success.";
              }
            }
            beg = end;
          }
        }
        return Status::Ok();}, 4));
  LOG(INFO) << "Finish load variables.";
  return Status::Ok();
}

Status CheckpointUtils::SaveVariables(
    size_t id,
    const std::string& checkpoint_path,
    const std::unordered_map<std::string, std::unique_ptr<Variable>>& vars,
    size_t timeout) {
  std::map<std::string, VariableInfo> dest_infos;
  for (auto&& item : infos_.infos) {
    dest_infos[item.name] = item;
  }
  if (vars.size() == 0) {
    return Status::Ok();
  }
  std::atomic<size_t> counter(vars.size());
  Status status = Status::Ok();
  std::promise<bool> ok;
  for (auto&& item : vars) {
    std::string name = item.first;
    ThreadPool::Global()->Schedule([&, name] {
          auto iter = dest_infos.find(name);
          if (iter == dest_infos.end()) {
            CK_CHECK_STATUS(status, Status::ArgumentError("Can't find variable[" + name + "] in variable_infos."), counter, ok);
          }
          VariableInfo info = iter->second;
          if (info.args["save"] == "false") {
            if (--counter == 0) {
              ok.set_value(true);
            }
            return;
          }
          int part = -1;
          for (size_t i = 0; i < info.parts.size(); i++) {
            if (info.parts[i].server == id) {
              part = i;
              break;
            }
          }
          if (part == -1) {
            CK_CHECK_STATUS(status, Status::ArgumentError("Not found variable[" + name + "] part[" + std::to_string(id) + "] in variable_infos when save variable."),
                            counter, ok);
          }
          VariableStruct vs;
          CK_CHECK_STATUS(status, VariableToStruct(vars.at(name), &vs), counter, ok);
          CK_CHECK_STATUS(status, SaveVariable(checkpoint_path, iter->first, part, &vs), counter, ok);
          if (--counter == 0) {
            ok.set_value(true);
          }
        });
  }
  std::future_status fstatus = ok.get_future().wait_for(std::chrono::minutes(timeout));
  if (fstatus != std::future_status::ready) {
    LOG(FATAL) << "Save checkpoint timeout, killing myself...";
    throw std::runtime_error("Save checkpoint timeout");
  }
  return status;
}

std::string CheckpointUtils::VariableNameToFileName(const std::string& name, size_t id) {
  std::string ret = name;
  for (auto& c : ret) {
    if (c == '/') {
      c = '$';
    }
  }
  return ret + '^' + std::to_string(id);
}

std::string CheckpointUtils::VariableInfoToFileName(const VariableInfo& info, size_t id) {
  auto p = info.args.find(VariableInfo::ORIGIN_FILE_PATH);
  if (p == info.args.end()) {
    LOG(ERROR) << info.name << " args not contain " << VariableInfo::ORIGIN_FILE_PATH;
    return "";
  }
  std::string file_path = p->second;
  auto n = info.args.find(VariableInfo::ORIGIN_NAME);
  std::string name = n == info.args.end() ? info.name : n->second;
  return file_path + "/" + VariableNameToFileName(name, id);
}

Status CheckpointUtils::MergeLoadVariable(const std::string& name, const VariableInfo& info, size_t beg, size_t end, std::unique_ptr<Variable>* result_variable) {
  std::vector<std::unique_ptr<LoadVariableStruct>> variables;
  size_t part_beg = 0;
  std::chrono::time_point<std::chrono::system_clock> time_start, time_end;
  time_start = std::chrono::system_clock::now();
  
  for (size_t i = 0; i < info.parts.size(); i++) {
    size_t part_end = part_beg + info.parts[i].size;
    if (part_beg < end && beg < part_end) {
      LOG(INFO) << name << ", part_beg [" << part_beg << "] part_end [" << part_end << "]";
      variables.emplace_back(new LoadVariableStruct);
      LoadVariableStruct& lvs = *variables.back();
      lvs.beg = part_beg;
      lvs.end = part_end;
      lvs.clip_beg = std::max(part_beg, beg);
      lvs.clip_end = std::min(part_end, end);
      PS_CHECK_STATUS(LoadVariable(info, i, &lvs.variable));
      if (!lvs.variable.initialized) {
        variables.pop_back();
      }
    }
    part_beg = part_end;
  }
  if (variables.size() == 0) {
    return Status::NotFound("Not found variable when load " + info.name);
  }
  time_end = std::chrono::system_clock::now();  
  LOG(INFO) << info.name << ", LoadVariable takes " << std::chrono::duration_cast<std::chrono::seconds>(time_end-time_start).count();

  VariableStruct var;
  // convert index_slicer
  if (info.type == VariableInfo::Type::kIndex) {
      std::unique_ptr<Data> slicer;
      TensorShape shape = variables[0]->variable.data.Shape();
      if (shape.Dims().size() != 0) {
        shape.Set(0, end - beg);
      }
      var.index_slicer = beg;
      slicer.reset(new WrapperData<size_t>(beg));
      var.data = Tensor(variables[0]->variable.data.Type(), shape, variables[0]->variable.data.GetInitializer()->Clone());
      size_t slice_size = SizeOfType(var.data.Type());
      if (shape.Dims().size() != 0) {
        slice_size = SizeOfType(var.data.Type()) * shape.NumElements() / shape[0];
      }
      for (const auto& lvs : variables) {
        if (lvs->beg <= beg) {
          QuickMemcpy(var.data.Raw<char>(), lvs->variable.data.Raw<char>() + (beg-lvs->beg)* slice_size, (lvs->clip_end-lvs->clip_beg) * slice_size);
        } else {
          QuickMemcpy(var.data.Raw<char>() + (lvs->beg-beg) * slice_size, lvs->variable.data.Raw<char>(), (lvs->clip_end-lvs->clip_beg) * slice_size);
        }
      }
      var.type = variables[0]->variable.type;
      for (auto& iter : variables[0]->variable.slots) {
          if (iter.second.joiner == Variable::SlotJoiner::kAnyOne) {
            var.slots[iter.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(*iter.second.tensor)), .joiner = iter.second.joiner};
          } else {
            Tensor* t = iter.second.tensor.get();
            TensorShape s = t->Shape();
            if (s.Dims().size() != 0) {
              s.Set(0, end - beg);
            }
            var.slots[iter.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(t->Type(), s, t->GetInitializer()->Clone())), .joiner = iter.second.joiner};
            size_t ssize = SizeOfType(t->Type());
            if (s.Dims().size() != 0) {              
              ssize = SizeOfType(t->Type()) * s.NumElements() / s[0];
            }
            for (auto& lvs : variables) {
              if (lvs->beg <= beg) {
                QuickMemcpy(var.slots[iter.first].tensor->Raw<char>(), lvs->variable.slots[iter.first].tensor->Raw<char>() + (beg-lvs->beg)* ssize, (lvs->clip_end-lvs->clip_beg) * ssize);
              } else {
                QuickMemcpy(var.slots[iter.first].tensor->Raw<char>()+(lvs->beg-beg) * slice_size, lvs->variable.slots[iter.first].tensor->Raw<char>(), (lvs->clip_end-lvs->clip_beg) * ssize);
              }
            }
          }
      }
      var.initialized = true;
      result_variable->reset(new Variable(new Tensor(var.data), slicer.release(), name));
      (*result_variable)->SetSlots(CloneSlots(var.slots));
      return Status::Ok();
  } else {
    time_start = std::chrono::system_clock::now();
    LoadHashVariable(variables, name, info, beg, end, *result_variable);
    time_end = std::chrono::system_clock::now();
    LOG(INFO) << info.name << ", load hash variable total, takes " << std::chrono::duration_cast<std::chrono::seconds>(time_end-time_start).count();
    return Status::Ok();
  }
}

Status CheckpointUtils::LoadHashVariable(const std::vector<std::unique_ptr<LoadVariableStruct>>& variables, const std::string& name, const VariableInfo& info, size_t beg, size_t end, std::unique_ptr<Variable>& result_variable) {
  auto time_start = std::chrono::system_clock::now();
  std::vector<std::vector<int64_t> > keys, values;
  size_t max_size = CalMaxSize(variables, name, beg, end, &keys, &values);
  if (max_size == 0) {
    max_size = 1;
  }
  auto time_end = std::chrono::system_clock::now();
  LOG(INFO) << name << ", CalMaxSize, takes " << std::chrono::duration_cast<std::chrono::seconds>(time_end-time_start).count() << ", max_size is " << max_size;

  time_start = std::chrono::system_clock::now();
  const Tensor& t = variables[0]->variable.data;
  TensorShape data_shape = t.Shape();
  data_shape.Set(0, max_size);
  HashMap* hashmap;
  if (info.type == VariableInfo::Type::kHash128) {
    hashmap = new HashMapImpl<Hash128Key>(100);
  } else {
    hashmap = new HashMapImpl<int64_t>(100);
  }
  Variable* var = new Variable(new Tensor(t.Type(), data_shape, t.GetInitializer()->Clone(), Tensor::TType::kSegment, false), new WrapperData<std::unique_ptr<HashMap> >(hashmap), name);
  std::unordered_map<std::string, Variable::Slot> slots;
  for (const auto& iter : variables[0]->variable.slots) {
    if (iter.second.joiner == Variable::SlotJoiner::kAnyOne) {
      slots[iter.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(*iter.second.tensor)), .joiner = iter.second.joiner};
    } else {
      Tensor& tt = *iter.second.tensor;
      TensorShape tt_shape = tt.Shape();
      tt_shape.Set(0, max_size);
      slots[iter.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(tt.Type(), tt_shape, tt.GetInitializer()->Clone(), Tensor::TType::kSegment, false)), .joiner = iter.second.joiner};
    }
  }
  time_end = std::chrono::system_clock::now();
  LOG(INFO) << name << ", initialize takes " << std::chrono::duration_cast<std::chrono::seconds>(time_end-time_start).count();

  for (size_t i = 0; i < variables.size(); i++) {
    const std::unique_ptr<LoadVariableStruct>& lvs = variables[i];
    std::vector<int64_t>& key = keys[i];
    std::vector<int64_t>& value = values[i];
    std::vector<size_t> ids;
    time_start = std::chrono::system_clock::now();
    size_t no_use;
    hashmap->Get((int64_t*)&key[0], value.size(), false, 1.0, &ids, nullptr, &no_use, 10000000000L);
    size_t slice_size = SizeOfType(var->GetData()->Type()) * var->GetData()->Shape().NumElements() / var->GetData()->Shape()[0];
    for (size_t j = 0; j < ids.size(); j++) {
      char* target = var->GetData()->Raw<char>(ids[j]);
      char* source = lvs->variable.data.Raw<char>(value[j]);
      memcpy(target, source, slice_size);
      for (auto& slot : slots) {
        if (slot.second.joiner == Variable::SlotJoiner::kVariableLike) {
          size_t ssize = SizeOfType(slot.second.tensor->Type()) * slot.second.tensor->Shape().NumElements() / slot.second.tensor->Shape()[0];
          char* target = slot.second.tensor->Raw<char>(ids[j]);
          char* source = lvs->variable.slots[slot.first].tensor->Raw<char>(value[j]);
          memcpy(target, source, ssize);
        }
      }
    }
    time_end = std::chrono::system_clock::now();
    LOG(INFO) << name << ", memcpy takes " << std::chrono::duration_cast<std::chrono::seconds>(time_end-time_start).count();
  }
  var->SetSlots(std::move(slots));
  result_variable.reset(var);
  return Status::Ok();
}

int64_t CheckpointUtils::CalMaxSize(const std::vector<std::unique_ptr<LoadVariableStruct> >& variables, const std::string& name, size_t begin, size_t end, std::vector<std::vector<int64_t> >* keys, std::vector<std::vector<int64_t> >* values) {
  keys->resize(variables.size());
  values->resize(variables.size());  
  for (size_t i = 0; i < variables.size(); i++) {
    if (variables[i]->variable.type == VariableStruct::kHashSlicer128) {
      (*keys)[i].reserve(variables[i]->variable.hash_slicer128.items.size() * 2);
      (*values)[i].reserve(variables[i]->variable.hash_slicer128.items.size());
    } else {
      (*keys)[i].reserve(variables[i]->variable.hash_slicer64.items.size());
      (*values)[i].reserve(variables[i]->variable.hash_slicer64.items.size());
    }
  }
  for (size_t i = 0; i < variables.size(); i++) {
    const std::unique_ptr<LoadVariableStruct>& lvs = variables[i];
    if (lvs->variable.type == VariableStruct::kHashSlicer128) {
      for (size_t j = 0; j < lvs->variable.hash_slicer128.items.size(); j++) {
        const HashMapItem<Hash128Key>& item = lvs->variable.hash_slicer128.items[j];
        uint32_t range = Hasher::Hash128(item.key.hash1, item.key.hash2);
        if (begin <= range && range < end) {
          (*keys)[i].push_back(item.key.hash1);
          (*keys)[i].push_back(item.key.hash2);
          (*values)[i].push_back(item.id);
        }
      }
    } else {
      for (size_t j = 0; j < lvs->variable.hash_slicer64.items.size(); j++) {
        const HashMapItem<int64_t>& item = lvs->variable.hash_slicer64.items[j];
        uint32_t range = Hasher::Hash64(item.key);
        if (begin <= range && range < end) {
          (*keys)[i].push_back(item.key);
          (*values)[i].push_back(item.id);
        }
      }
    }
  }
  size_t total = 0;
  for (size_t i = 0; i < variables.size(); i++) {
    total += (*values)[i].size();
  }
  LOG(INFO) << name << ", variables.size() " << variables.size() << " begin " << begin << " end " << end << " total " << total;
  return total;
}

Status CheckpointUtils::LoadVariable(const VariableInfo& info, size_t part, VariableStruct* var) {
  std::unique_ptr<FileSystem::ReadStream> s;
  Status st = FileSystem::OpenReadStreamAny(VariableInfoToFileName(info, part), &s);
  if (!st.IsOk()) {
    LOG(ERROR) << "Open " << VariableInfoToFileName(info, part) << " failed.";
    var->initialized = false;
    return st;
  }
  return LoadVariable(info.name + " part[" + std::to_string(part) + "]", s.get(), var);
}

Status CheckpointUtils::SaveVariable(const std::string& checkpoint, const std::string& var_name, size_t part, VariableStruct* var) {
  std::unique_ptr<FileSystem::WriteStream> s;
  PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(checkpoint + '/' + VariableNameToFileName(var_name, part), &s));
  return SaveVariable(s.get(), var);
}

Status CheckpointUtils::VariableToStruct(const std::unique_ptr<Variable>& var, VariableStruct* vs) {
  Data* slicer = var->GetSlicer();
  if (dynamic_cast<WrapperData<size_t>*>(slicer) != nullptr) {
    vs->type = VariableStruct::kIndexSlicer;
    vs->index_slicer = dynamic_cast<WrapperData<size_t>*>(slicer)->Internal();
  } else if (dynamic_cast<WrapperData<std::unique_ptr<HashMap> >*>(slicer) != nullptr) {
    HashMap* hashmap = dynamic_cast<WrapperData<std::unique_ptr<HashMap> >*>(slicer)->Internal().get();
    if (dynamic_cast<HashMapImpl<int64_t>*>(hashmap) != nullptr) {
      vs->type = VariableStruct::kHashSlicer64;
      dynamic_cast<HashMapImpl<int64_t>*>(hashmap)->GetItems(&vs->hash_slicer64);
    } else if (dynamic_cast<HashMapImpl<Hash128Key>*>(hashmap) != nullptr) {
      vs->type = VariableStruct::kHashSlicer128;
      dynamic_cast<HashMapImpl<Hash128Key>*>(hashmap)->GetItems(&vs->hash_slicer128);
    }
  } else {
    return Status::NotImplemented("Not Implemented variable slicer type");
  }
  vs->data = *var->GetData();
  vs->slots = CloneSlots(var->GetSlots());
  vs->initialized = true;
  return Status::Ok();
}

Status CheckpointUtils::LoadVariable(const std::string& name, FileSystem::ReadStream* s, VariableStruct* var) {
  PS_CHECK_STATUS(s->ReadRaw(&(var->type)));
  switch (var->type) {
  case VariableStruct::kIndexSlicer:
    PS_CHECK_STATUS(s->ReadRaw(&(var->index_slicer)));
    break;
  case VariableStruct::kHashSlicer128:
    PS_CHECK_STATUS(s->ReadRaw(&(var->hash_slicer128.count)));
    PS_CHECK_STATUS(s->ReadTBBVec(&(var->hash_slicer128.items)));
    break;
  case VariableStruct::kHashSlicer64:
    PS_CHECK_STATUS(s->ReadRaw(&(var->hash_slicer64.count)));
    PS_CHECK_STATUS(s->ReadTBBVec(&(var->hash_slicer64.items)));
    break;
  default:
    return Status::NotImplemented("Not Implemented variable slicer type");
  }
  PS_CHECK_STATUS(LoadTensor(name, s, var->type, &var->data));
  size_t slot_size;
  PS_CHECK_STATUS(s->ReadRaw(&slot_size));
  for (size_t i = 0; i < slot_size; i++) {
    std::string slot_name;
    PS_CHECK_STATUS(s->ReadStr(&slot_name));
    Variable::Slot& slot = var->slots[slot_name];
    slot.tensor.reset(new Tensor);
    PS_CHECK_STATUS(s->ReadRaw(&slot.joiner));
    PS_CHECK_STATUS(LoadTensor(name + " slot[" + slot_name + "]", s, var->type, slot.tensor.get()));
  }
  var->initialized = true;
  return Status::Ok();
}

Status CheckpointUtils::SaveVariable(FileSystem::WriteStream* s, VariableStruct* var) {
  PS_CHECK_STATUS(s->WriteRaw(var->type));
  switch (var->type) {
  case VariableStruct::kIndexSlicer:
    PS_CHECK_STATUS(s->WriteRaw(var->index_slicer));
    break;
  case VariableStruct::kHashSlicer128:
    PS_CHECK_STATUS(s->WriteRaw(var->hash_slicer128.count.load()));
    PS_CHECK_STATUS(s->WriteTBBVec(var->hash_slicer128.items));
    break;
  case VariableStruct::kHashSlicer64:
    PS_CHECK_STATUS(s->WriteRaw(var->hash_slicer64.count.load()));
    PS_CHECK_STATUS(s->WriteTBBVec(var->hash_slicer64.items));
    break;
  default:
    return Status::NotImplemented("Not Implemented variable slicer type");
  }
  PS_CHECK_STATUS(SaveTensor(s, var->data));
  size_t slot_size = var->slots.size();
  PS_CHECK_STATUS(s->WriteRaw(slot_size));
  for (auto&& slot : var->slots) {
    PS_CHECK_STATUS(s->WriteStr(slot.first));
    PS_CHECK_STATUS(s->WriteRaw(slot.second.joiner));
    PS_CHECK_STATUS(SaveTensor(s, *slot.second.tensor));
  }
  return Status::Ok();
}

Status CheckpointUtils::LoadTensor(const std::string& name, FileSystem::ReadStream* s, VariableStruct::SlicerType slicer_type, Tensor* data) {
  DataType type;
  std::vector<size_t> shape;
  Initializer* initializer;
  size_t initializer_type;
  std::string initializer_buf;

  PS_CHECK_STATUS(s->ReadRaw(&type));
  PS_CHECK_STATUS(s->ReadVec(&shape));
  PS_CHECK_STATUS(s->ReadRaw(&initializer_type));
  PS_CHECK_STATUS(s->ReadStr(&initializer_buf));
  size_t len;
  serializer::MemGuard mem;
  serializer::Fragment frag(&initializer_buf[0], initializer_buf.size());
  PS_CHECK_STATUS(serializer::DeserializeAny<Initializer>(initializer_type, &frag, 0, &initializer, &len, mem));
  Tensor result(type, TensorShape(shape), initializer, Tensor::TType::kContinuous, false);
  PS_CHECK_STATUS(s->Read(result.Raw<char>(), result.Shape().NumElements() * SizeOfType(type)));
  *data = result;
  return Status::Ok();
}

Status CheckpointUtils::SaveTensor(FileSystem::WriteStream* s, const Tensor& data) {
  DataType type = data.Type();
  TensorShape tensor_shape = data.Shape();
  const std::vector<size_t>& shape = tensor_shape.Dims();
  Initializer* initializer = data.GetInitializer();
  size_t initializer_type;
  std::string initializer_buf;

  std::vector<serializer::Fragment> frags;
  serializer::MemGuard mem;
  PS_CHECK_STATUS(serializer::SerializeAny<Initializer>(initializer, &initializer_type, &frags, mem));
  for (auto frag : frags) {
    initializer_buf.append(frag.base, frag.size);
  }

  PS_CHECK_STATUS(s->WriteRaw(type));
  PS_CHECK_STATUS(s->WriteVec(shape));
  PS_CHECK_STATUS(s->WriteRaw(initializer_type));
  PS_CHECK_STATUS(s->WriteStr(initializer_buf));
  if (data.TensorType() == Tensor::TType::kContinuous) { 
    PS_CHECK_STATUS(s->Write(data.Raw<char>(), tensor_shape.NumElements() * SizeOfType(type)));
  } else if (data.TensorType() == Tensor::TType::kSegment) {
    size_t slice_size = tensor_shape.NumElements()/tensor_shape[0];
    for (size_t i = 0; i < tensor_shape[0] / data.SegmentSize(); i++) {
      PS_CHECK_STATUS(s->Write(data.Raw<char>(i * data.SegmentSize()), data.SegmentSize() * slice_size * SizeOfType(type)));
    }
  } else {
    return Status::ArgumentError("Tensor type not support .");
  }
  return Status::Ok();
}

std::unordered_map<std::string, Variable::Slot> CheckpointUtils::CloneSlots(const std::unordered_map<std::string, Variable::Slot>& slots) {
  std::unordered_map<std::string, Variable::Slot> ret;
  for (auto&& item : slots) {
    ret[item.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(*item.second.tensor)), .joiner = item.second.joiner};
  }
  return std::move(ret);
}


}
}

