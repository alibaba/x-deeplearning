/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#include "xdl/core/lib/status.h"
#include "xdl/core/utils/time_utils.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"
#include "xdl/core/utils/string_utils.h"

#include <ps-plus/common/file_system.h>
#include <ps-plus/common/string_utils.h>
#define private public
#include <ps-plus/server/checkpoint_utils.h>
#include <ps-plus/scheduler/scheduler_impl.h>
#undef private
#include <ps-plus/message/variable_info.h>
#include <ps-plus/common/data.h>
#include <ps-plus/common/serializer.h>
#include <stdexcept>

namespace xdl {

class PsConvertCkptVariableOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("with_slots", &with_slots_));
    XDL_CHECK_STATUS(ctx->GetAttr("convert_index_with_enter", &convert_index_with_enter_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    uint64_t start_time = TimeUtils::NowMicros();
    Tensor t_ckpt_dir, t_output_dir, t_variables;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &t_ckpt_dir), done);
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &t_output_dir), done);
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(2, &t_variables), done);
    std::string ckpt_dir = t_ckpt_dir.Scalar<std::string>();
    if (ckpt_dir.back() != '/') {
      ckpt_dir += "/";
    }
    std::string output_dir = t_output_dir.Scalar<std::string>();
    std::string s_variables = t_variables.Scalar<std::string>();
    std::string ckpt;
    XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(GetLatestCheckpoint(ckpt_dir, &ckpt)), done);
    std::vector<ps::VariableInfo> infos;
    size_t server_num;
    XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(ps::scheduler::SchedulerImpl::ReadVariableInfoMeta(ckpt, &server_num, &infos)), done);
    for (auto& item : infos) {
      item.args[ps::VariableInfo::ORIGIN_FILE_PATH] = ckpt;
    }
    ps::server::CheckpointUtils utils(ps::VariableInfoCollection{.infos=infos});
    std::vector<std::string> variables = ps::StringUtils::split(s_variables, ",");
    Status::ErrorCode error_code = Status::kOk;
    #pragma omp parallel for
    for (size_t i = 0; i < variables.size(); ++i) {
      std::string& variable = variables[i];
      for(auto info : infos) {
        if (info.name == variable) {
          Status st = PS2XDL::ConvertStatus(ConvertVariable(info, utils, output_dir, with_slots_, convert_index_with_enter_));
          if (!st.IsOk()) {
            error_code = st.Code();
            printf("ErrorMsg: %s\n", st.Msg().c_str());
          }
        }
      }
    }
    Status status = error_code == Status::kOk ? Status::Ok()
                                              : Status(error_code, "ps_convert_ckpt_variable_op failed.");
    done(status);
    printf("finish convert ckpt[%s], duration[%dms]\n", ckpt_dir.c_str(), (TimeUtils::NowMicros() - start_time) / 1000);
  }

 private:    
  ps::Status GetLatestCheckpoint(const std::string& checkpoint_dir, std::string* checkpoint) {
    std::vector<std::string> checkpoints;
    PS_CHECK_STATUS(ps::scheduler::SchedulerImpl::ReadCheckpoints(checkpoint_dir, true, &checkpoints));
    if (checkpoints.size() == 0) {
      *checkpoint = checkpoint_dir;
      printf("Converting [%s]\n", checkpoint->c_str());
    } else {
      *checkpoint = checkpoint_dir + "/" + checkpoints.back();
      printf("GetLatestCheckpoint [%s]\n", checkpoint->c_str());
    }
    return ps::Status::Ok();
  }

  std::string ReplaceVariableName(const std::string& name) {
    std::string ret;
    for (char c: name) {
      if (c == '/') {
        ret.append(1, '&');
      } else {
        ret.append(1, c);
      }
    }
    return ret;
  }

  template<typename KeyType>
  std::string ToString(KeyType key) {
    return "";
  }

  ps::Status ConvertVariable(const ps::VariableInfo& info, ps::server::CheckpointUtils& utils, const std::string& output_dir,
                             bool with_slots=false, bool convert_index_with_enter=false) {
    ps::FileSystem::WriteStream* output_stream;
    std::string file_name = output_dir + "/" + ReplaceVariableName(info.name);
    ps::FileSystem::RemoveAny(file_name);
    PS_CHECK_STATUS(ps::FileSystem::OpenWriteStreamAny(file_name, &output_stream));
    std::map<std::string, ps::FileSystem::WriteStream*> slot_stream_map;
    for (size_t i = 0; i < info.parts.size(); i++) {
      ps::server::CheckpointUtils::VariableStruct vs;
      printf("Start convert [%s], part[%d]\n", info.name.c_str(), i);
      uint64_t start_time = TimeUtils::NowMicros();
      PS_CHECK_STATUS(utils.LoadVariable(info, i, &vs));
      if (!vs.initialized) {
        return ps::Status::DataLoss("Load variable " + info.name + " failed.");
      }
      if (i == 0 && with_slots) {
        for (const auto& iter : vs.slots) {
          ps::FileSystem::WriteStream* slot_stream;            
          std::string file_name = output_dir + "/" + ReplaceVariableName(info.name) + "&" + ReplaceVariableName(iter.first);
          ps::FileSystem::RemoveAny(file_name);
          PS_CHECK_STATUS(ps::FileSystem::OpenWriteStreamAny(file_name, &slot_stream));
          slot_stream_map[iter.first] = slot_stream;
        }
      }

      int sparse_id_cnt = 0;
      switch (vs.type) {
        case ps::server::CheckpointUtils::VariableStruct::kHashSlicer128: {
          sparse_id_cnt = vs.hash_slicer128.items.size();
          PS_CHECK_STATUS(ConvertHashVariable(vs, vs.hash_slicer128, output_stream, with_slots, slot_stream_map));
          break;
        }
        case ps::server::CheckpointUtils::VariableStruct::kHashSlicer64: {
          sparse_id_cnt = vs.hash_slicer64.items.size();
          PS_CHECK_STATUS(ConvertHashVariable(vs, vs.hash_slicer64, output_stream, with_slots, slot_stream_map));
          break;
        }          
        case ps::server::CheckpointUtils::VariableStruct::kIndexSlicer: {
          PS_CHECK_STATUS(ConvertIndexVariable(vs, output_stream, i, info.parts.size(), with_slots, convert_index_with_enter, slot_stream_map));
          break;
        }
        default:
          return ps::Status::NotImplemented("Not Implemented variable slicer type");
      }

      printf("finish convert [%s], part[%d], id_cnt[%d], duration[%dms]\n", 
             info.name.c_str(), i, 
             sparse_id_cnt, 
             (TimeUtils::NowMicros() - start_time) / 1000);      
    }
    for (auto& iter : slot_stream_map) {
      iter.second->Close();  
    }
    output_stream->Close();
    return ps::Status::Ok();
  }

  template<typename KeyType>
  ps::Status ConvertHashVariable(ps::server::CheckpointUtils::VariableStruct& vs, const ps::HashMapStruct<KeyType>& slicer, ps::FileSystem::WriteStream* output_stream, bool with_slots, std::map<std::string, ps::FileSystem::WriteStream*>& slot_stream_map) {
    std::vector<size_t> dims = vs.data.Shape().Dims();
    size_t slicer_size = 1;
    for (size_t dim = 1; dim < dims.size(); dim++) {
      slicer_size *= dims[dim];
    }

    

    for (size_t i = 0; i < slicer.items.size(); i++) {
      const ps::HashMapItem<KeyType>& item = slicer.items[i];
      CASES(vs.data.Type(),
        do {
          std::string ss;
          T* raw = vs.data.Raw<T>(item.id);
          ss += ToString<KeyType>(item.key);
          for (size_t j = 0; j < slicer_size; j++) {
            ss += "," + StringUtils::ToStringPrecision(*raw++);
          }
          ss += "\n";
          output_stream->Write(ss.c_str(), ss.size());
        } while (0));
      if (with_slots) {
        for (const auto& iter: vs.slots) {
          CASES(iter.second.tensor->Type(),
                do {
                  std::string slot_stream;
                  if (iter.second.joiner == ps::server::Variable::SlotJoiner::kVariableLike) {
                    std::vector<size_t> sdims = iter.second.tensor->Shape().Dims();
                    size_t ssize = 1;
                    for (size_t dim = 1; dim < sdims.size(); dim++) {
                      ssize *= sdims[dim];
                    }
                    T* raw = iter.second.tensor->Raw<T>(item.id);
                    slot_stream += ToString<KeyType>(item.key);
                    for (size_t j = 0; j < ssize; j++) {
                      slot_stream += "," + StringUtils::ToStringPrecision(*raw++);
                    }
                    slot_stream += "\n";
                  } else {
                    // for kAnyOne slot, only print first one
                    if (i == 0) {
                      T* raw = iter.second.tensor->Raw<T>();
                      slot_stream += StringUtils::ToStringPrecision(*raw++);
                      for (size_t j = 1; j < iter.second.tensor->Shape().NumElements(); j++) {
                        slot_stream += "," + StringUtils::ToStringPrecision(*raw++);
                      }
                    }
                  }
                  slot_stream_map[iter.first]->Write(slot_stream.c_str(), slot_stream.size());
                } while (0));
        }
      }
    }
    return ps::Status::Ok();
  }

  ps::Status ConvertIndexVariable(ps::server::CheckpointUtils::VariableStruct& vs, ps::FileSystem::WriteStream* output_stream, int index, int total,
                                  bool with_slots, bool convert_index_with_enter, std::map<std::string, ps::FileSystem::WriteStream*>& slot_stream_map) {
    std::string ss;
    CASES(vs.data.Type(), do {
      if (vs.data.Shape().Size() == 0) {
        std::string value = StringUtils::ToStringPrecision(vs.data.Raw<T>()[0]);
        value += "\n";
        output_stream->Write(value.c_str(), value.size());        
      } else {
        if (vs.data.Shape().Size() > 2) {
          printf("Warning: vs.data.Shape().Size() = %ld\n", vs.data.Shape().Size());
        }
        for (int i = 0; i < vs.data.Shape().NumElements(); ++i) {
          if (i > 0 || index != 0) {
            ss += (convert_index_with_enter && vs.data.Shape().Size() == 2 && i % vs.data.Shape()[1] == 0) ? "\n" : ",";
          }
          ss += StringUtils::ToStringPrecision(vs.data.Raw<T>()[i]);
        }
        if (index == total - 1) {
          ss += "\n";
        }
        output_stream->Write(ss.c_str(), ss.size());
      }
    } while (0));
    if (with_slots) {
      for (const auto& iter: vs.slots) {
        std::string slot_stream;
        CASES(iter.second.tensor->Type(), do {
          if (iter.second.tensor->Shape().Size() == 0) {
            std::string value = StringUtils::ToStringPrecision(iter.second.tensor->Raw<T>()[0]);
            value += "\n";
            slot_stream_map[iter.first]->Write(value.c_str(), value.size());
          } else {
            for (int i = 0; i < iter.second.tensor->Shape().NumElements(); ++i) {
              if (i > 0 || index != 0) {
                slot_stream += ",";
              }
              slot_stream += StringUtils::ToStringPrecision(iter.second.tensor->Raw<T>()[i]);
            }
          }
        } while (0));
        if (index == total - 1) {
          slot_stream += "\n";
        }
        slot_stream_map[iter.first]->Write(slot_stream.c_str(), slot_stream.size());
      }
    }
    return ps::Status::Ok();
  }

  bool with_slots_;    
  bool convert_index_with_enter_;
};

template<>  
std::string PsConvertCkptVariableOp::ToString<int64_t>(int64_t key) {
  return std::to_string(key);
}

template<>  
std::string PsConvertCkptVariableOp::ToString<ps::Hash128Key>(ps::Hash128Key key) {
  return std::to_string(key.hash1) + "," + std::to_string(key.hash2);
}

XDL_DEFINE_OP(PsConvertCkptVariableOp)
   .Input("checkpoint_dir", DataType::kInt8)
   .Input("output_dir", DataType::kInt8)
   .Input("variables", DataType::kInt8)
   .Attr("with_slots", AttrValue::kBool, false)
   .Attr("convert_index_with_enter", AttrValue::kBool, false);

XDL_REGISTER_KERNEL(PsConvertCkptVariableOp, PsConvertCkptVariableOp).Device("CPU");


} // namespace xdl



