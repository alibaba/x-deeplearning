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

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

#include <ps-plus/common/file_system.h>
#include <ps-plus/common/string_utils.h>
#include <ps-plus/server/checkpoint_utils.h>
#include <ps-plus/message/variable_info.h>
#include <ps-plus/common/data.h>
#include <ps-plus/common/serializer.h>

namespace xdl {

class PsConvertCkptVariableOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("hash64", &hash64_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
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
    XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(LoadMeta(ckpt, &infos)), done);
    ps::server::CheckpointUtils utils(ckpt, ps::VariableInfoCollection{.infos=infos});
    std::vector<std::string> variables = ps::StringUtils::split(s_variables, ",");
    for (std::string variable: variables) {
      for(auto info : infos) {
        if (info.name == variable) {
          XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(ConvertVariable(info, utils, output_dir)), done);
        }
      }
    }
    done(Status::Ok());
  }

 private:    
  ps::Status GetLatestCheckpoint(const std::string& checkpoint_dir, std::string* checkpoint) {
    std::unique_ptr<ps::FileSystem::ReadStream> s;
    ps::Status st = ps::FileSystem::OpenReadStreamAny(checkpoint_dir + "checkpoints", &s);
    if (!st.IsOk()) {
        *checkpoint = checkpoint_dir;
        printf("Converting [%s]\n", checkpoint->c_str());        
        return ps::Status::Ok();
    }
    size_t size;
    PS_CHECK_STATUS(s->ReadRaw(&size));
    if (size == 0) {
      return ps::Status::ArgumentError("Read checkpoints size is 0");
    }
    for (size_t i = 0; i < size; i++) {
      PS_CHECK_STATUS(s->ReadStr(checkpoint));
    }
    *checkpoint = checkpoint_dir + *checkpoint + "/";
    printf("GetLatestCheckpoint [%s]\n", checkpoint->c_str());
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

  ps::Status ConvertVariable(const ps::VariableInfo& info, ps::server::CheckpointUtils& utils, const std::string& output_dir) {
    ps::FileSystem::WriteStream* output_stream;
    std::string file_name = output_dir + "/" + ReplaceVariableName(info.name);
    ps::FileSystem::RemoveAny(file_name);
    PS_CHECK_STATUS(ps::FileSystem::OpenWriteStreamAny(file_name, &output_stream));
    for (size_t i = 0; i < info.parts.size(); i++) {
      ps::server::CheckpointUtils::VariableStruct vs;
      printf("Start convert [%s], part[%ld]\n", info.name.c_str(), i);
      PS_CHECK_STATUS(utils.LoadVariable(info.name, i, &vs));
      if (!vs.initialized) {
        return ps::Status::DataLoss("Load variable " + info.name + " failed.");
      }
      switch (vs.type) {
        case ps::server::CheckpointUtils::VariableStruct::kHashSlicer: {
          PS_CHECK_STATUS(ConvertHashSparseVariable(vs, output_stream));
          continue;
        }
        case ps::server::CheckpointUtils::VariableStruct::kIndexSlicer: {
          PS_CHECK_STATUS(ConvertIndexVariable(vs, output_stream));
          continue;
        }
        default:
          return ps::Status::NotImplemented("Not Implemented variable slicer type");
      }
    }
    output_stream->Close();
    return ps::Status::Ok();
  }

  ps::Status LoadMeta(const std::string& checkpoint_path, std::vector<ps::VariableInfo>* infos) {
    std::unique_ptr<ps::FileSystem::ReadStream> s;
    PS_CHECK_STATUS(ps::FileSystem::OpenReadStreamAny(checkpoint_path  + "__meta__", &s));
    size_t server_count;

    {
      size_t infos_type;
      std::string infos_buf;
      PS_CHECK_STATUS(s->ReadRaw(&server_count));
      PS_CHECK_STATUS(s->ReadRaw(&infos_type));
      PS_CHECK_STATUS(s->ReadStr(&infos_buf));
      ps::Data* info_wrapper;
      size_t len;
      ps::serializer::MemGuard mem;
      ps::serializer::Fragment frag(&infos_buf[0], infos_buf.size());
      PS_CHECK_STATUS(ps::serializer::DeserializeAny<ps::Data>(infos_type, &frag, 0, &info_wrapper, &len, mem));
      std::unique_ptr<ps::Data> info_wrapper_deleter(info_wrapper);
      ps::WrapperData<ps::VariableInfoCollection>* info_wrapper_converted = dynamic_cast<ps::WrapperData<ps::VariableInfoCollection>*>(info_wrapper);
      if (info_wrapper_converted == nullptr) {
        return ps::Status::Unknown("Variable Info Load Error");
      }
      *infos = info_wrapper_converted->Internal().infos;
    }
    return ps::Status::Ok();
  }

  ps::Status ConvertHashSparseVariable(ps::server::CheckpointUtils::VariableStruct& vs, ps::FileSystem::WriteStream* output_stream) {
    std::vector<size_t> dims = vs.data.Shape().Dims();
    size_t slicer_size = 1;
    for (size_t dim = 1; dim < dims.size(); dim++) {
      slicer_size *= dims[dim];
    }
    for (size_t i = 0; i < vs.hash_slicer.items.size(); i++) {
      ps::HashMapItem& item = vs.hash_slicer.items[i];
      CASES(vs.data.Type(),
        do {
          std::stringstream ss;
          T* raw = vs.data.Raw<T>();
          if (!hash64_) {
            ss << std::to_string(item.x) << ",";
          }
          ss << std::to_string(item.y);
          for (size_t j = 0; j < slicer_size; j++) {
            ss << "," << std::to_string(raw[item.id * slicer_size + j]);
          }
          ss << std::endl;
          output_stream->Write(ss.str().c_str(), ss.str().size());
        } while (0));
    }
    return ps::Status::Ok();
  }

  ps::Status ConvertIndexVariable(ps::server::CheckpointUtils::VariableStruct& vs, ps::FileSystem::WriteStream* output_stream) {
    CASES(vs.data.Type(), do {
      if (vs.data.Shape().Size() == 0) {
        std::string value = std::to_string(vs.data.Raw<T>()[0]);
        value += "\n";
        output_stream->Write(value.c_str(), value.size());        
      } else {
        std::stringstream ss;
        for (int i = 0; i < vs.data.Shape().NumElements(); ++i) {
          if (i > 0) {
            ss << ",";
          } 
          ss << std::to_string(vs.data.Raw<T>()[i]);
        }

        ss << std::endl;
        output_stream->Write(ss.str().c_str(), ss.str().size());
      }
    } while (0));

    return ps::Status::Ok();
  }

  bool hash64_;
};

XDL_DEFINE_OP(PsConvertCkptVariableOp)
   .Input("checkpoint_dir", DataType::kInt8)
   .Input("output_dir", DataType::kInt8)
   .Input("variables", DataType::kInt8)
   .Attr("hash64", AttrValue::kBool, false);

XDL_REGISTER_KERNEL(PsConvertCkptVariableOp, PsConvertCkptVariableOp).Device("CPU");

} // namespace xdl



