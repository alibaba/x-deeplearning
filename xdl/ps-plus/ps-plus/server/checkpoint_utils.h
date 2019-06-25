#ifndef PS_PLUS_SERVER_CHECKPOINT_UTILS_H_
#define PS_PLUS_SERVER_CHECKPOINT_UTILS_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/file_system.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/server/variable.h"

#include <string>

namespace ps {
namespace server {

class CheckpointUtils {
 public:
  CheckpointUtils(const std::string& path, const VariableInfoCollection& infos);
  Status LoadVariables(
      const VariableInfoCollection& infos,
      size_t id,
      std::unordered_map<std::string, std::unique_ptr<Variable>>* vars);
  Status SaveVariables(
      size_t id,
      const std::unordered_map<std::string, std::unique_ptr<Variable>>& vars);
  struct VariableStruct {
    enum SlicerType : int32_t {
      kIndexSlicer = 0,
      kHashSlicer = 1
    };
    bool initialized;
    SlicerType type;
    HashMap::HashMapStruct hash_slicer;
    size_t index_slicer;
    Tensor data;
    std::unordered_map<std::string, Variable::Slot> slots;
  };
  Status LoadVariable(const std::string& var_name, size_t part, VariableStruct* var);
 private:
  struct LoadVariableStruct {
    VariableStruct variable;
    size_t beg, end;
    size_t clip_beg, clip_end;
  };
  Status MergeLoadVariable(const std::string& var_name, const VariableInfo& info, size_t beg, size_t end, VariableStruct* var, std::unique_ptr<Variable>* result_variable);
  Status SaveVariable(const std::string& var_name, size_t part, VariableStruct* var);
  Status SaveVariableExt(const std::string &var_name, VariableStruct *var, size_t part);
  Status SaveSparseVariableBinary(const std::string &var_name, VariableStruct *var, size_t part);
  int64_t CalMaxSize(const std::vector<std::unique_ptr<LoadVariableStruct>>& variables, size_t begin, size_t end);
  static std::string VariableNameToFileName(const std::string& name, size_t id);
  static Status StructToVariable(const VariableStruct& vs, std::unique_ptr<Variable>* var, const VariableInfo& info, size_t part);
  static Status VariableToStruct(const std::unique_ptr<Variable>& var, VariableStruct* vs);
  static Status LoadVariable(FileSystem::ReadStream* s, VariableStruct* var);
  static Status SaveVariable(FileSystem::WriteStream* s, VariableStruct* var);
  static Status LoadTensor(FileSystem::ReadStream* s, Tensor* data);
  static Status SaveTensor(FileSystem::WriteStream* s, const Tensor& data);
  static std::unordered_map<std::string, Variable::Slot> CloneSlots(const std::unordered_map<std::string, Variable::Slot>& slots);

  std::string path_;
  std::unordered_map<std::string, VariableInfo> infos_;
};

}
}

#endif

