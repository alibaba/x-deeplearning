#include <glog/logging.h>
#include "ps-plus/server/checkpoint_utils.h"
#include "ps-plus/common/serializer.h"
#include "ps-plus/common/hasher.h"
#include "ps-plus/common/types.h"
#include "xdl/core/utils/logging.h"
#include <sys/time.h>
#include <string>
#include <iostream>
#include <thread>
#include <vector>
namespace ps {
namespace server {

CheckpointUtils::CheckpointUtils(const std::string& path, const VariableInfoCollection& infos) {
  path_ = path;
  for (auto&& item : infos.infos) {
    infos_[item.name] = item;
  }
}

/*
Status CheckpointUtils::LoadVariables(
    const VariableInfoCollection& infos,
    size_t id,
    std::unordered_map<std::string, std::unique_ptr<Variable>>* vars) {
  ThreadPool* pool = ThreadPool::Global();
  std::promise<Status> status;
  std::atomic<size_t> counter(0);
  for (auto&& info : infos.infos) {
    for (auto& part : info.parts) {
      if (part.server == id) {
        counter++;
      }
    }
  }
  for (auto&& info : infos.infos) {
    auto iter = infos_.find(info.name);
    if (iter == infos_.end()) {
      continue;
    }
    size_t beg = 0;
    for (size_t i = 0; i < info.parts.size(); i++) {
      auto part = info.parts[i];
      size_t end = beg + part.size;
      if (part.server == id) {
        LOG_INFO("Loading variable[%s] part[%d].", info.name.c_str(), i);
        (*vars)[info.name] = std::unique_ptr<Variable>(nullptr);
        VariableInfo vi = iter->second;
        pool->Schedule([vi, beg, end, vars, i, this, &counter, &status] {
                                VariableStruct vs;
                                Status st = MergeLoadVariable(vi.name, vi, beg, end, &vs, &(*vars)[vi.name]);
                                if (!st.IsOk()) {
                                  status.set_value(st);
                                }
                                LOG(INFO) << "Load variable" << vi.name.c_str() << " part " << i << " MergeLoadVariable success.";
                                if (--counter == 0) {
                                  status.set_value(Status::Ok());
                                }
                            });
      }
      beg = end;
    }
  }
  infos_.clear();
  for (auto&& item : infos.infos) {
    infos_[item.name] = item;
  }
  std::future<Status> future = status.get_future();
  future.wait();
  PS_CHECK_STATUS(future.get());
  LOG(INFO) << "Finish load variables.";
  return Status::Ok();
}
*/

Status CheckpointUtils::LoadVariables(
    const VariableInfoCollection& infos,
    size_t id,
    std::unordered_map<std::string, std::unique_ptr<Variable>>* vars) {
  for (auto&& info : infos.infos) {
    auto iter = infos_.find(info.name);
    if (iter == infos_.end()) {
      continue;
    }
    size_t beg = 0;
    for (size_t i = 0; i < info.parts.size(); i++) {
      auto part = info.parts[i];
      size_t end = beg + part.size;
      if (part.server == id) {
//          LOG(DEBUG) << "Loading variable" << info.name.c_str() << " part" << i;
        VariableInfo vi = iter->second;
        VariableStruct vs;
        Status st = MergeLoadVariable(vi.name, vi, beg, end, &vs, &(*vars)[vi.name]);
        if (st.Code() == Status::ErrorCode::kNotFound) {
          LOG(WARNING) << st.Msg();
          vars->erase(vi.name);
        } else {
          PS_CHECK_STATUS(st);
//          LOG(DEBUG) << "Load variable" << vi.name << " part" << i << " MergeLoadVariable success.";
        }
      }
      beg = end;
    }
  }
  infos_.clear();
  for (auto&& item : infos.infos) {
    infos_[item.name] = item;
  }
//  LOG(DEBUG) << "Finish load variables.";
  return Status::Ok();
}

Status CheckpointUtils::SaveVariables(
    size_t id,
    const std::unordered_map<std::string, std::unique_ptr<Variable>>& vars) {
  for (auto&& item : vars) {
    auto iter = infos_.find(item.first);
    if (iter == infos_.end()) {
      LOG(ERROR) << "Can't find variable " << item.first << " in variable_infos.";
      continue;
    }
    VariableInfo info = iter->second;
    if (info.args["save"] == "false") {
      LOG(ERROR) << "Variable" << item.first << " not to save.";
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
      LOG(ERROR) << "Not found variable" << item.first << " part" << id << " in variable_infos when save variable.";
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

Status CheckpointUtils::MergeLoadVariable(const std::string& var_name, const VariableInfo& info, size_t beg, size_t end, VariableStruct* var, std::unique_ptr<Variable>* result_variable) {
  std::vector<std::unique_ptr<LoadVariableStruct>> variables;
  size_t part_beg = 0;
  clock_t time_start, time_end;
  time_start = clock();

  for (size_t i = 0; i < info.parts.size(); i++) {
    size_t part_end = part_beg + info.parts[i].size;
//    LOG(DEBUG) << "part_beg " << part_beg << " part_end" << part_end;
    if (part_beg < end && beg < part_end) {
      variables.emplace_back(new LoadVariableStruct);
      LoadVariableStruct& lvs = *variables.back();
      lvs.beg = part_beg;
      lvs.end = part_end;
      lvs.clip_beg = std::max(part_beg, beg);
      lvs.clip_end = std::min(part_end, end);
      PS_CHECK_STATUS(LoadVariable(var_name, i, &lvs.variable));
      if (!lvs.variable.initialized) {
        variables.pop_back();
      }
    }
    part_beg = part_end;
  }
  if (variables.size() == 0) {
    return Status::NotFound("Not found variable when load " + info.name);
  }
  time_end = clock();

  // convert index_slicer
  if (info.type == VariableInfo::Type::kIndex) {
      std::unique_ptr<Data> slicer;
      TensorShape shape = variables[0]->variable.data.Shape();
      if (shape.Dims().size() != 0) {
        shape.Set(0, end - beg);
      }
      var->index_slicer = beg;
      slicer.reset(new WrapperData<size_t>(beg));
      var->data = Tensor(variables[0]->variable.data.Type(), shape, variables[0]->variable.data.GetInitializer()->Clone(), true);
      size_t slice_size = SizeOfType(var->data.Type());
      if (shape.Dims().size() != 0) {
        slice_size = SizeOfType(var->data.Type()) * shape.NumElements() / shape[0];
      }
      for (const auto& lvs : variables) {
        if (lvs->beg <= beg) {
          QuickMemcpy(var->data.Raw<char>(), lvs->variable.data.Raw<char>() + (beg-lvs->beg)* slice_size, (lvs->clip_end-lvs->clip_beg) * slice_size);
        } else {
          QuickMemcpy(var->data.Raw<char>() + (lvs->beg-beg) * slice_size, lvs->variable.data.Raw<char>(), (lvs->clip_end-lvs->clip_beg) * slice_size);
        }
      }
      var->type = variables[0]->variable.type;
      for (auto& iter : variables[0]->variable.slots) {
          if (iter.second.joiner == Variable::SlotJoiner::kAnyOne) {
              var->slots[iter.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(*iter.second.tensor)), .joiner = iter.second.joiner};
          } else {
              Tensor* t = iter.second.tensor.get();
              TensorShape s = t->Shape();
              if (s.Dims().size() != 0) {              
                  s.Set(0, end - beg);
              }
              var->slots[iter.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(t->Type(), s, t->GetInitializer()->Clone(), true)), .joiner = iter.second.joiner};
              size_t ssize = SizeOfType(t->Type());
              if (s.Dims().size() != 0) {              
                  ssize = SizeOfType(t->Type()) * s.NumElements() / s[0];
              }
              for (auto& lvs : variables) {
                  if (lvs->beg <= beg) {
                      QuickMemcpy(var->slots[iter.first].tensor->Raw<char>(), lvs->variable.slots[iter.first].tensor->Raw<char>() + (beg-lvs->beg)* ssize,
                              (lvs->clip_end-lvs->clip_beg) * ssize);
                  } else {
                      QuickMemcpy(var->slots[iter.first].tensor->Raw<char>()+(lvs->beg-beg) * slice_size, lvs->variable.slots[iter.first].tensor->Raw<char>(),
                              (lvs->clip_end-lvs->clip_beg) * ssize);
                  }
              }
          }
      }
      var->initialized = true;
      result_variable->reset(new Variable(new Tensor(var->data), slicer.release()));
      (*result_variable)->SetSlots(CloneSlots(var->slots));
      return Status::Ok();
  }

  time_start = clock();
  // convert hash_slicer
  size_t max_size = CalMaxSize(variables, beg, end);
  var->hash_slicer.counter = max_size;
//  LOG(DEBUG) << "variable " << info.name << " slice " << max_size;
  TensorShape data_shape = variables[0]->variable.data.Shape();
  max_size = int(max_size * 1.2) + 10;
  data_shape.Set(0, max_size);
  var->data = Tensor(variables[0]->variable.data.Type(), data_shape, variables[0]->variable.data.GetInitializer()->Clone(), true);
  for (const auto& iter : variables[0]->variable.slots) {
      if (iter.second.joiner == Variable::SlotJoiner::kAnyOne) {
          var->slots[iter.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(*iter.second.tensor)), .joiner = iter.second.joiner};
      } else {
          Tensor* t = iter.second.tensor.get();
          TensorShape s = t->Shape();
          s.Set(0, max_size);
          var->slots[iter.first] = Variable::Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(t->Type(), s, t->GetInitializer()->Clone(), true)), .joiner = iter.second.joiner};
      }
  }
  time_end = clock();

  time_start = clock();  
  std::unique_ptr<Data> slicer;
  std::unique_ptr<WrapperData<HashMap>> xslicer(new WrapperData<HashMap>(max_size));
  std::vector<int64_t> ids, reused_ids;
  for (const auto& iter: variables) {
    const HashMap::HashMapStruct& hash_slicer = iter->variable.hash_slicer;
    std::vector<int64_t> keys, item_ids;
    for (size_t index = 0; index < hash_slicer.items.size(); index++) {
      auto& item = hash_slicer.items[index];
      uint32_t range = Hasher::Hash128(item.x, item.y);
      if (beg <= range && range < end) {
        keys.push_back(item.x);
        keys.push_back(item.y);
        item_ids.push_back(item.id);
      }
      if (keys.size() > 400000 || index == hash_slicer.items.size()-1) {
        if (xslicer->Internal().Get(&keys[0], keys.size()/2, 2, &ids, &reused_ids) != 0) {
          return Status::ArgumentError("insert hashmap failed.");
        }
        size_t slice_size = SizeOfType(var->data.Type()) * var->data.Shape().NumElements() / var->data.Shape()[0];
        for (size_t ki = 0; ki < keys.size()/2; ki++) {
          CASES(var->data.Type(), do {
                      T* target = var->data.Raw<T>();
                      T* source = iter->variable.data.Raw<T>();
                      memcpy((void*)target + (ids[ki]*slice_size), (void*)source + (item_ids[ki]*slice_size), slice_size);
                  } while(0));
          for (auto& slot : var->slots) {
            if (slot.second.tensor->Shape().IsScalar()) continue;
            size_t ssize = SizeOfType(slot.second.tensor->Type()) * slot.second.tensor->Shape().NumElements() / slot.second.tensor->Shape()[0];
            CASES(slot.second.tensor->Type(), do {
                        T* target = slot.second.tensor->Raw<T>();
                        T* source = iter->variable.slots[slot.first].tensor->Raw<T>();
                        memcpy((void*)target + (ids[ki]*ssize), (void*)source + (item_ids[ki]*ssize), ssize);
                    } while(0));
          }
        }
        keys.clear();
        item_ids.clear();
      }
    }
  }
  time_end = clock();
  
  //slicer.reset(xslicer.release());
  result_variable->reset(new Variable(new Tensor(var->data), xslicer.release()));
  (*result_variable)->SetSlots(CloneSlots(var->slots));
  var->type = VariableStruct::kHashSlicer;
  var->initialized = true;
  return Status::Ok();
}

int64_t CheckpointUtils::CalMaxSize(const std::vector<std::unique_ptr<LoadVariableStruct>>& variables, size_t begin, size_t end) {
  int64_t total = 0;
  for (auto& lvs : variables) {
    if (begin <= lvs->beg && lvs->end <= end) {
      total += lvs->variable.hash_slicer.items.size();
      continue;
    }
    for (auto item : lvs->variable.hash_slicer.items) {
      uint32_t range = Hasher::Hash128(item.x, item.y);
      if (begin <= range && range < end) {
        ++total;
      }
    }
  }
  return total;
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

Status CheckpointUtils::LoadVariable(const std::string& var_name, size_t part, VariableStruct* var) {
  std::unique_ptr<FileSystem::ReadStream> s;
  Status st = FileSystem::OpenReadStreamAny(path_ + '/' + VariableNameToFileName(var_name, part), &s);
  if (!st.IsOk()) {
    LOG(ERROR) << "Open" << path_ << "/" << VariableNameToFileName(var_name, part) << " failed.";
    var->initialized = false;
    return Status::Ok();
  }
  return LoadVariable(s.get(), var);
}

Status CheckpointUtils::SaveVariable(const std::string& var_name, size_t part, VariableStruct* var) {
  std::unique_ptr<FileSystem::WriteStream> s;
  auto raw_file_name = VariableNameToFileName(var_name, part);
  PS_CHECK_STATUS(FileSystem::OpenWriteStreamAny(path_ + '/' + raw_file_name, &s));
  PS_CHECK_STATUS(SaveVariable(s.get(), var));
  
  return SaveVariableExt(var_name, var, part);
}

Status CheckpointUtils::SaveVariableExt(const std::string &var_name, VariableStruct *var, size_t part){
  switch (var->type) {
  case VariableStruct::kHashSlicer:{
    PS_CHECK_STATUS(SaveSparseVariableBinary(var_name, var, part));
    break;
  default:
    break;
  }
  }
  return Status::Ok();
}

struct Piece {
  unsigned long key;
  float val[0];
};

static void SaveSparseVariableBinaryThread(const std::string &path,
        const std::string &var_name, 
        CheckpointUtils::VariableStruct *var,
        size_t thread_id,
        size_t total_threads){

  if(thread_id >= var->hash_slicer.items.size()){
    return;
  }
  std::unique_ptr<FileSystem::WriteStream> s;
  auto raw_file_name = var_name + "_" + std::to_string(thread_id);
  auto status = FileSystem::OpenWriteStreamAny(path + '/' + raw_file_name, &s);
  if(!status.IsOk()){
    std::cout <<"open " << path << "/" << raw_file_name << " failed\n";
    return;
  }
  std::vector<size_t> dims = var->data.Shape().Dims();
  size_t slicer_size = 1;
  for (size_t dim = 1; dim < dims.size(); dim++) {
    slicer_size *= dims[dim];
  }

  std::unique_ptr<char[]> buf;
  auto piece_size = sizeof(unsigned long) + sizeof(float) * slicer_size;
  buf = std::make_unique<char[]>(piece_size);
  auto piece = reinterpret_cast<Piece*>(buf.get());
  for (size_t i = thread_id; i < var->hash_slicer.items.size(); i += total_threads) {
    ps::HashMapItem& item = var->hash_slicer.items[i];
    CASES(var->data.Type(),
    do {
      T* raw = var->data.Raw<T>();
      piece->key = item.y;
      for (size_t j = 0; j < slicer_size; j++) {
        piece->val[j] = raw[item.id * slicer_size + j];
      }
      s->Write(buf.get(), piece_size);
    } while (0));
  }
  s->Close();
}
Status CheckpointUtils::SaveSparseVariableBinary(const std::string &var_name, VariableStruct *var, size_t part){
  struct timeval ts0, ts1;
  gettimeofday(&ts0, NULL);

  std::vector<std::thread*> threads;
  size_t thread_num = 10;
  auto file_name = "emb_bin/" + VariableNameToFileName(var_name, part);
  for(size_t i = 0; i < thread_num; ++i){
    threads.push_back(new std::thread(SaveSparseVariableBinaryThread, path_,
                file_name, var, i, thread_num));
  }

  for(size_t i = 0; i < thread_num; ++i){
    threads[i]->join();
    delete threads[i];
  }
  gettimeofday(&ts1, NULL);

  std::cout << "save var " << var_name << " "  << part  << " "
      << std::to_string(var->hash_slicer.items.size()) << " " << std::to_string(ts0.tv_sec) 
      << " " << std::to_string(ts1.tv_sec) << "\n";
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
//    LOG(DEBUG) << "index_slicer size is " << var->index_slicer;
    break;
  case VariableStruct::kHashSlicer:
    PS_CHECK_STATUS(s->ReadRaw(&(var->hash_slicer.counter)));
//    LOG(DEBUG) << "Hash_slicer counter is " << var->hash_slicer.counter;
    PS_CHECK_STATUS(s->ReadVec(&(var->hash_slicer.items)));
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

}
}

