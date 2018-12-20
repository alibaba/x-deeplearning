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

#include "ps-plus/server/checkpoint_utils.h"

#include <glog/logging.h>

#include "ps-plus/common/serializer.h"
#include "ps-plus/common/hasher.h"

namespace ps {
namespace server {

CheckpointUtils::CheckpointUtils(const std::string& path, const VariableInfoCollection& infos) {
  path_ = path;
  for (auto&& item : infos.infos) {
    infos_[item.name] = item;
  }
}

Status CheckpointUtils::LoadVariables(
    const VariableInfoCollection& infos,
    size_t id,
    std::unordered_map<std::string, std::unique_ptr<Variable>>* vars) {
  for (auto&& info : infos.infos) {
    auto iter = infos_.find(info.name);
    if (iter == infos_.end()) {
      continue;
    }
    VariableStruct vs;
    bool found = false;
    size_t beg = 0;
    for (size_t i = 0; i < info.parts.size(); i++) {
      auto part = info.parts[i];
      size_t end = beg + part.size;
      if (part.server == id) {
        LOG(INFO) << "Loading variable[" << info.name << "] part[" << id << "].";
        PS_CHECK_STATUS(MergeLoadVariable(info.name, iter->second, beg, end, &vs));
        LOG(INFO) << "Load variable[" << info.name << "] part[" << id << "] MergeLoadVariable success.";
        found = true;
        if (!vs.initialized) {
          LOG(ERROR) << "Load variable[" << info.name << "] part[" << id << "] failed.";
          break;
        }
        PS_CHECK_STATUS(StructToVariable(vs, &(*vars)[info.name], info, i));
        LOG(INFO) << "Load variable[" << info.name << "] part[" << id << "] structToVariable success.";
        break;
      }
      beg = end;
    }
    if (!found) {
      LOG(ERROR) << "Not found variable[" << info.name << "] part[" << id << "] in variable_infos when load variable.";
    }
  }
  return Status::Ok();
}

Status CheckpointUtils::SaveVariables(
    size_t id,
    const std::unordered_map<std::string, std::unique_ptr<Variable>>& vars) {
  for (auto&& item : vars) {
    auto iter = infos_.find(item.first);
    if (iter == infos_.end()) {
      LOG(ERROR) << "Can't find variable[" << item.first << "] in variable_infos.";
      continue;
    }
    VariableInfo info = iter->second;
    if (info.args["save"] == "false") {
      LOG(ERROR) << "Variable[" << item.first << "] not to save.";
      continue;
    }
    int part = -1;
    for (size_t i = 0; i < info.parts.size(); i++) {
      if (info.parts[i].server == id) {
        part = i;
        break;
      }
    }
    if (part == -1) {
      LOG(ERROR) << "Not found variable[" << item.first << "] part[" << id << 
                "] in variable_infos when save variable.";
      continue;
    }
    VariableStruct vs;
    PS_CHECK_STATUS(VariableToStruct(item.second, &vs));
    PS_CHECK_STATUS(SaveVariable(iter->first, part, &vs));
  }
  return Status::Ok();
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

Status CheckpointUtils::MergeLoadVariable(const std::string& var_name, const VariableInfo& info, size_t beg, size_t end, VariableStruct* var) {
  std::vector<std::unique_ptr<LoadVariableStruct>> variables;
  size_t part_beg = 0;
  for (size_t i = 0; i < info.parts.size(); i++) {
    size_t part_end = part_beg + info.parts[i].size;
    if (part_beg < end && beg < part_end) {
      variables.emplace_back(new LoadVariableStruct);
      LoadVariableStruct& lvs = *variables.back();
      lvs.beg = part_beg;
      lvs.end = part_end;
      lvs.clip_beg = std::max(part_beg, beg);
      lvs.clip_end = std::min(part_end, end);
      PS_CHECK_STATUS(LoadVariable(var_name, i, &lvs.variable));
    }
    part_beg = part_end;
  }
  // A Shortcut for unchanged info
  if (variables.size() == 1 && variables[0]->clip_beg == variables[0]->beg && variables[0]->clip_end == variables[0]->end) {
    *var = std::move(variables[0]->variable);
    return Status::Ok();
  }
  //TODO
  return Status::NotImplemented("changed variable info is not supported");
}

Status CheckpointUtils::LoadVariable(const std::string& var_name, size_t part, VariableStruct* var) {
  std::unique_ptr<FileSystem::ReadStream> s;
  Status st = FileSystem::OpenReadStreamAny(path_ + '/' + VariableNameToFileName(var_name, part), &s);
  if (!st.IsOk()) {
    LOG(ERROR) << "Open " << path_ << "/" << VariableNameToFileName(var_name, part) << " failed.";
    var->initialized = false;
    return Status::Ok();
  }
  return LoadVariable(s.get(), var);
}

Status CheckpointUtils::SaveVariable(const std::string& var_name, size_t part, VariableStruct* var) {
  std::unique_ptr<FileSystem::WriteStream> s;
  PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(path_ + '/' + VariableNameToFileName(var_name, part), &s));
  return SaveVariable(s.get(), var);
}

Status CheckpointUtils::StructToVariable(const VariableStruct& vs, std::unique_ptr<Variable>* var, const VariableInfo& info, size_t part) {
  std::unique_ptr<Data> slicer;
  switch (vs.type) {
  case VariableStruct::kIndexSlicer: {
    slicer.reset(new WrapperData<size_t>(vs.index_slicer));
    break;
  }
  case VariableStruct::kHashSlicer: {
    if (info.shape.empty()) {
      return Status::ArgumentError("CheckpointUtils: Hash Shape should not be scalar");
    }
    size_t hashmap_size = std::max(info.shape[0] * info.parts[part].size / Hasher::kTargetRange, vs.hash_slicer.counter) + 10;
    std::unique_ptr<WrapperData<HashMap>> xslicer(new WrapperData<HashMap>(hashmap_size));
    xslicer->Internal().SetHashKeys(vs.hash_slicer);
    slicer.reset(xslicer.release());
    break;
  }
  default:
    return Status::NotImplemented("Not Implemented variable slicer type");
  }
  var->reset(new Variable(new Tensor(vs.data), slicer.release()));
  (*var)->SetSlots(CloneSlots(vs.slots));
  return Status::Ok();
}

Status CheckpointUtils::VariableToStruct(const std::unique_ptr<Variable>& var, VariableStruct* vs) {
  Data* slicer = var->GetSlicer();
  if (dynamic_cast<WrapperData<size_t>*>(slicer) != nullptr) {
    vs->type = VariableStruct::kIndexSlicer;
    vs->index_slicer = dynamic_cast<WrapperData<size_t>*>(slicer)->Internal();
  } else if (dynamic_cast<WrapperData<HashMap>*>(slicer) != nullptr) {
    vs->type = VariableStruct::kHashSlicer;
    int ret = dynamic_cast<WrapperData<HashMap>*>(slicer)->Internal().GetHashKeys(&vs->hash_slicer);
    if (ret) {
      return Status::Unknown("HashMap GetHashKeys Internal Error");
    }
  } else {
    return Status::NotImplemented("Not Implemented variable slicer type");
  }
  vs->data = *var->GetData();
  vs->slots = CloneSlots(var->GetSlots());
  vs->initialized = true;
  return Status::Ok();
}

Status CheckpointUtils::LoadVariable(FileSystem::ReadStream* s, VariableStruct* var) {
  PS_CHECK_STATUS(s->ReadRaw(&(var->type)));
  switch (var->type) {
  case VariableStruct::kIndexSlicer:
    PS_CHECK_STATUS(s->ReadRaw(&(var->index_slicer)));
    break;
  case VariableStruct::kHashSlicer:
    PS_CHECK_STATUS(s->ReadRaw(&(var->hash_slicer.counter)));
    LOG(INFO) << "Hash_slicer counter is " << var->hash_slicer.counter;
    PS_CHECK_STATUS(s->ReadVec(&(var->hash_slicer.items)));
    LOG(INFO) << "Hash_slicer items size is " << var->hash_slicer.items.size();   
    break;
  default:
    return Status::NotImplemented("Not Implemented variable slicer type");
  }
  PS_CHECK_STATUS(LoadTensor(s, &var->data));
  size_t slot_size;
  PS_CHECK_STATUS(s->ReadRaw(&slot_size));
  for (size_t i = 0; i < slot_size; i++) {
    std::string slot_name;
    PS_CHECK_STATUS(s->ReadStr(&slot_name));
    Variable::Slot& slot = var->slots[slot_name];
    slot.tensor.reset(new Tensor);
    PS_CHECK_STATUS(s->ReadRaw(&slot.joiner));
    PS_CHECK_STATUS(LoadTensor(s, slot.tensor.get()));
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
  case VariableStruct::kHashSlicer:
    PS_CHECK_STATUS(s->WriteRaw(var->hash_slicer.counter));
    PS_CHECK_STATUS(s->WriteVec(var->hash_slicer.items));
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

Status CheckpointUtils::LoadTensor(FileSystem::ReadStream* s, Tensor* data) {
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
  Tensor result(type, TensorShape(shape), initializer, false);
  PS_CHECK_STATUS(s->Read(result.Raw<char>(), result.Shape().NumElements() * SizeOfType(type)));
  *data = result;
  return Status::Ok();
}

Status CheckpointUtils::SaveTensor(FileSystem::WriteStream* s, const Tensor& data) {
  DataType type = data.Type();
  const std::vector<size_t>& shape = data.Shape().Dims();
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
  PS_CHECK_STATUS(s->Write(data.Raw<char>(), data.Shape().NumElements() * SizeOfType(type)));

  return Status::Ok();
}

std::unordered_map<std::string, Variable::Slot> CheckpointUtils::CloneSlots(const std::unordered_map<std::string, Variable::Slot>& slots) {
  std::unordered_map<std::string, Variable::Slot> ret;
  for (auto&& item : slots) {
    ret[item.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(*item.second.tensor)), .joiner = item.second.joiner};
  }
  return std::move(ret);
}

// used for local server, which load all variables to one server
Status CheckpointUtils::LoadVariables(
    const VariableInfoCollection& infos,
    std::unordered_map<std::string, std::unique_ptr<Variable>>* vars) {
  for (auto&& info : infos.infos) {
    std::vector<std::unique_ptr<VariableStruct> > vss;
    for (size_t i = 0; i < info.parts.size(); i++) {
      vss.emplace_back(new VariableStruct());
      PS_CHECK_STATUS(LoadVariable(info.name, i, vss.back().get()));
    }

    if (vss[0]->type == VariableStruct::kIndexSlicer) {
      PS_CHECK_STATUS(VaribaleStructsToVariable(vss, &(*vars)[info.name]));
    } else if (vss.size() == 1) {
      StructToVariable(*(vss[0]), &(*vars)[info.name], info, 0);
    } else {
      return Status::NotImplemented("hash feature only support load from local model checkpoint!");
    }
  }

  return Status::Ok();
}

Status CheckpointUtils::VaribaleStructsToVariable(const std::vector<std::unique_ptr<VariableStruct> >& vss, 
                                                  std::unique_ptr<Variable>* var) {
  if (vss.empty()) return Status::ArgumentError("no VariableStruct found");
  std::unique_ptr<Data> slicer;
  switch (vss[0]->type) {
  case VariableStruct::kIndexSlicer: 
    slicer.reset(new WrapperData<size_t>(0));
    break;
  // TODO: support hash feature
  case VariableStruct::kHashSlicer:
    return Status::NotImplemented("not support hash feature yet!");
    break;
  default:
    return Status::NotImplemented("Not Implemented variable slicer type");
  }

  Tensor* data = new Tensor();
  std::unordered_map<std::string, Variable::Slot> slots;
  PS_CHECK_STATUS(MergeVariableAndSlots(vss, data, &slots));
  var->reset(new Variable(data, slicer.release()));
  (*var)->SetSlots(std::move(slots));
  return Status::Ok();
}

Status CheckpointUtils::MergeVariableAndSlots(const std::vector<std::unique_ptr<VariableStruct> >& vss,
                                              Tensor* data,
                                              std::unordered_map<std::string, Variable::Slot>* slots) {
  std::vector<Tensor*> datas;
  std::unordered_map<std::string, std::vector<Variable::Slot*> > grouped_slots;
  for (auto& vs: vss) {
    datas.push_back(const_cast<Tensor*>(&(vs->data)));
    for (auto& item: vs->slots) {
      grouped_slots[item.first].emplace_back(const_cast<Variable::Slot*>(&item.second));
    }
  }

  PS_CHECK_STATUS(MergeTensors(datas, data));
  PS_CHECK_STATUS(MergeSlots(grouped_slots, slots));
  return Status::Ok();
}

Status CheckpointUtils::MergeTensors(const std::vector<Tensor*>& tensors,
                                     Tensor* result) {
  if (tensors.empty()) return Status::Ok();
  if (!IsCompatible(tensors)) return Status::ArgumentError("can't merge compatible tensors");
  int64_t total_size = 0;
  std::vector<int64_t> buf_sizes;
  std::vector<TensorShape> shapes;
  for (auto& t: tensors) {
    CASES(t->Type(), do {
      buf_sizes.push_back(t->Shape().NumElements() * sizeof(T));
      total_size += buf_sizes.back();
      shapes.emplace_back(t->Shape());
    } while (0));
  }

  char* tensor_buf = new char[total_size];
  int64_t offset = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    memcpy(tensor_buf + offset, tensors[i]->Raw<char*>(), buf_sizes[i]);
    offset += buf_sizes[i];
  }

  TensorShape merged_shape;
  MergeTensorShape(shapes, &merged_shape);
  *result = Tensor(tensors[0]->Type(), merged_shape, tensor_buf, nullptr);
  return Status::Ok();
}

bool CheckpointUtils::IsCompatible(const std::vector<Tensor*>& tensors) {
  if (tensors.empty()) return true;
  for (size_t i = 1; i < tensors.size(); ++i) {
    if (tensors[0]->Shape().Size() != tensors[i]->Shape().Size()) {
      return false;
    }

    if (tensors[0]->Shape().Size() > 0 && 
        (tensors[0]->Shape()[0] != tensors[i]->Shape()[0])) {
      return false;
    }
  }

  return true;
}

void CheckpointUtils::MergeTensorShape(const std::vector<TensorShape>& shapes,
                                       TensorShape* shape) {
  if (shapes[0].Size() == 0) {
    *shape = TensorShape({});
  } else {
    size_t dim0 = 0;
    std::vector<size_t> new_dims;
    for (auto& s: shapes) {
      dim0 += s[0];
    }

    new_dims.push_back(dim0);
    new_dims.insert(new_dims.end(), shapes[0].Dims().begin() + 1, shapes[0].Dims().end());
    *shape = TensorShape(new_dims);
  }
}

Status CheckpointUtils::MergeSlots(const std::unordered_map<std::string, std::vector<Variable::Slot*> >& slots,
                                   std::unordered_map<std::string, Variable::Slot>* merged_slots) {
  for (auto& item: slots) {
    std::vector<Tensor*> datas;
    Tensor merged_data;
    for (auto& slot: item.second) {
      datas.push_back(slot->tensor.get());
      PS_CHECK_STATUS(MergeTensors(datas, &merged_data));
      (*merged_slots)[item.first] = 
        Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(merged_data)), 
                       .joiner = slot->joiner};
    }
  }

  return Status::Ok();
}

}
}

