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

#ifndef PS_PLUS_SERVER_CHECKPOINT_UTILS_H_
#define PS_PLUS_SERVER_CHECKPOINT_UTILS_H_

#include <memory>
#include <string>
#include "ps-plus/common/status.h"
#include "ps-plus/common/file_system.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/server/variable.h"
#include "ps-plus/common/hasher.h"

namespace ps {
namespace server {

class CheckpointUtils {
 public:
  CheckpointUtils(const VariableInfoCollection& infos);
  Status LoadVariables(
      const VariableInfoCollection& infos,
      size_t id,
      std::unordered_map<std::string, std::unique_ptr<Variable>>* vars);
  Status SaveVariables(
      size_t id,
      const std::string& checkpoint_path,
      const std::unordered_map<std::string, std::unique_ptr<Variable>>& vars,
      size_t timeout=30);

 private:
  struct VariableStruct {
    enum SlicerType : int32_t {
      kIndexSlicer = 0,
      kHashSlicer128 = 1,
      kHashSlicer64 = 2,
    };
    bool initialized;
    SlicerType type;
    HashMapStruct<Hash128Key> hash_slicer128;
    HashMapStruct<int64_t> hash_slicer64;
    size_t index_slicer;
    Tensor data;
    std::unordered_map<std::string, Variable::Slot> slots;
  };
  struct LoadVariableStruct {
    VariableStruct variable;
    size_t beg, end;
    size_t clip_beg, clip_end;
  };
  Status LoadVariable(const VariableInfo& info, size_t part, VariableStruct* var);
  Status VariableToStruct(const std::unique_ptr<Variable>& var, VariableStruct* vs);
  static Status SaveVariable(const std::string& checkpoint_path, const std::string& var_name, size_t part, VariableStruct* var);
  static std::string VariableInfoToFileName(const VariableInfo& info, size_t id);
  static std::string VariableNameToFileName(const std::string& name, size_t id);
  static Status LoadVariable(const std::string& name, FileSystem::ReadStream* s, VariableStruct* var);
  static Status SaveVariable(FileSystem::WriteStream* s, VariableStruct* var);
  static Status LoadTensor(const std::string& name, FileSystem::ReadStream* s, VariableStruct::SlicerType slicer_type, Tensor* data);
  static Status SaveTensor(FileSystem::WriteStream* s, const Tensor& data);
  static std::unordered_map<std::string, Variable::Slot> CloneSlots(const std::unordered_map<std::string, Variable::Slot>& slots);
  Status MergeLoadVariable(const std::string& name, const VariableInfo& info, size_t beg, size_t end, std::unique_ptr<Variable>* result_variable);
  Status LoadHashVariable(const std::vector<std::unique_ptr<LoadVariableStruct>>& variables, const std::string& name, const VariableInfo& info, size_t beg, size_t end, std::unique_ptr<Variable>& result_variable);
  static int64_t CalMaxSize(const std::vector<std::unique_ptr<LoadVariableStruct> >& variables, const std::string& name, size_t begin, size_t end, std::vector<std::vector<int64_t> >* keys, std::vector<std::vector<int64_t> >* values);
  VariableInfoCollection infos_;
};

}
}

#endif

