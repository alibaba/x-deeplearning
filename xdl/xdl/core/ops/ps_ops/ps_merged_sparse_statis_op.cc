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
#include "xdl/core/framework/cpu_device.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/utils/string_utils.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsMergedSparseStatisOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("statis_type", &statis_type_));
    std::string var_name_str;
    XDL_CHECK_STATUS(ctx->GetAttr("var_names", &var_name_str));
    var_names_ = StringUtils::split(var_name_str, ",");
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);

    std::vector<Tensor> ids;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("ids", &ids), done);
    std::vector<Tensor> indexs;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("indexs", &indexs), done);
    std::vector<Tensor> segments;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("segments", &segments), done);
    std::vector<Tensor> sample_indexs;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("sample_indexs", &sample_indexs), done);
    std::vector<Tensor> sample_segments;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("sample_segments", &sample_segments), done);
    std::vector<Tensor> labels;
    if (statis_type_ == "click") {
      XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("labels", &labels), done);
    }
    std::vector<Tensor> t_save_ratios;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInputList("save_ratios", &t_save_ratios), done);
    Tensor global_step;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput("global_step", &global_step), done);
    Tensor statis_decay;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput("statis_decay", &statis_decay), done);
    Tensor statis_decay_period;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput("statis_decay_period", &statis_decay_period), done);

    std::vector<Tensor> clicks;
    auto check = [](bool b) {
      return b ? Status::Ok()
          : Status::ArgumentError("Illegal check ps_merged_sparse_statis_op");
    };
    XDL_CHECK_STATUS_ASYNC(check(indexs.size() == ids.size()), done);
    XDL_CHECK_STATUS_ASYNC(check(segments.size() == ids.size()), done);
    XDL_CHECK_STATUS_ASYNC(check(sample_indexs.size() == ids.size()), done);
    XDL_CHECK_STATUS_ASYNC(check(sample_segments.size() == ids.size()), done);
    XDL_CHECK_STATUS_ASYNC(check(t_save_ratios.size() == ids.size()), done);
    if (statis_type_ == "click") {
      XDL_CHECK_STATUS_ASYNC(check(labels.size() == ids.size()), done);
    }

    clicks.resize(ids.size());
    #pragma omp parallel for
    for (size_t i = 0; i < ids.size(); ++i) {  // number of var
      int32_t last_segment = 0;
      const size_t bs = segments[i].Shape().NumElements();
      const size_t index_num = indexs[i].Shape().Dims()[0];
      const size_t uniq_num = ids[i].Shape().Dims()[0];
      size_t label_dim = 0;
      float* plabel = nullptr;                         // {bs,2}
      int32_t* psegment = segments[i].Raw<int32_t>();  // {bs}
      int32_t* psample_index = sample_indexs[i].Raw<int32_t>();      // {index_num}
      int32_t* psample_segment = sample_segments[i].Raw<int32_t>();  // {uniq_num}
      if (statis_type_ == "click") {
        //XDL_CHECK_STATUS_ASYNC(check(labels[i].Shape().Dims().size() <= 2), done);
        plabel = labels[i].Raw<float>();
        label_dim = labels[i].Shape().Size() == 1 ? 1 : labels[i].Shape().Dims()[1];
        //XDL_CHECK_STATUS_ASYNC(check(labels[i].Shape().Dims()[0] == bs), done);
      }
      //XDL_CHECK_STATUS_ASYNC(check(sample_indexs[i].Shape().Dims()[0] == index_num), done);
      //XDL_CHECK_STATUS_ASYNC(check(sample_segments[i].Shape().Dims()[0] == uniq_num), done);
      Tensor click(ctx->GetDevice(), TensorShape({uniq_num}), DataType::kInt32);
      int32_t* pclick = click.Raw<int32_t>();
      for (size_t j = 0; j < uniq_num; ++j) {
        int32_t nclick = 0;
        if (plabel == nullptr) {
          nclick = psample_segment[j] - (j == 0 ? 0 : psample_segment[j-1]);
        } else {
          for (int32_t k = j == 0 ? 0 : psample_segment[j-1]; k < psample_segment[j]; ++k) {
            float flabel = plabel[psample_index[k] * label_dim + label_dim - 1];
            nclick += flabel > 0.5 ? 1 : 0;
          }
        }
        pclick[j] = nclick;
      }
      clicks[i] = std::move(click);
    }

    std::vector<ps::Tensor> converted_ids;
    for (auto& id: ids) {
      converted_ids.emplace_back();
      XDL_CHECK_STATUS_ASYNC(
          XDL2PS::ConvertTensorZC(id, &converted_ids.back()),
          done);
    }

    std::vector<ps::Tensor> converted_clicks;
    for (auto& click: clicks) {
      converted_clicks.emplace_back();
      XDL_CHECK_STATUS_ASYNC(
          XDL2PS::ConvertTensorZC(click, &converted_clicks.back()),
          done);
    }

    ps::Tensor converted_global_step;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensorZC(global_step, &converted_global_step),
        done);

    ps::Tensor converted_statis_decay;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensorZC(statis_decay, &converted_statis_decay),
        done);

    ps::Tensor converted_statis_decay_period;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensorZC(statis_decay_period, &converted_statis_decay_period),
        done);

    std::vector<ps::Tensor>* ps_result = new std::vector<ps::Tensor>();
    auto cb = [ps_result, ctx, done](const ps::Status& st) {
      std::vector<Tensor> result;
      std::unique_ptr<std::vector<ps::Tensor> > result_deleter(ps_result);
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      for (auto& item: *ps_result) {
        result.emplace_back();
        XDL_CHECK_STATUS_ASYNC(
            PS2XDL::ConvertTensorZC(item, &result.back()),
            done);
      }
      ctx->SetOutputList("outputs", result);
      done(Status::Ok());
    };

    std::vector<float> save_ratios;
    for (size_t i = 0; i < t_save_ratios.size(); ++i) {
      save_ratios.push_back(t_save_ratios[i].Scalar<float>());
    }    
    if (var_type_ == VarType::kHash128 || var_type_ == VarType::kHash64) {
      client->MergedHashStatis(var_names_, converted_ids, save_ratios, converted_clicks, converted_global_step,
                               converted_statis_decay, converted_statis_decay_period,
                               statis_type_, ps_result, cb);
    } else {
      done(Status::ArgumentError("PsMergedSparseStatisOp var_type must be hash"));
    }
  }

 private:
  std::string var_name_;
  std::string statis_type_;
  VarType var_type_;
  std::vector<std::string> var_names_;
};

XDL_DEFINE_OP(PsMergedSparseStatisOp)
  .InputListV2("ids", "input_type")
  .InputListV2("indexs", "input_type_1")
  .InputListV2("segments", "input_type_2")
  .InputListV2("sample_indexs", "input_type_3")
  .InputListV2("sample_segments", "input_type_4")
  .InputListV2("labels", "input_type_5")
  .InputListV2("save_ratios", "input_type_6")
  .Input("global_step", DataType::kInt64)
  .Input("statis_decay", DataType::kDouble)
  .Input("statis_decay_period", DataType::kInt64)
  .OutputList("outputs", DataType::kFloat, "output_size")
  .Attr("input_type", AttrValue::kDataTypeList)
  .Attr("input_type_1", AttrValue::kDataTypeList)
  .Attr("input_type_2", AttrValue::kDataTypeList)
  .Attr("input_type_3", AttrValue::kDataTypeList)
  .Attr("input_type_4", AttrValue::kDataTypeList)
  .Attr("input_type_5", AttrValue::kDataTypeList)
  .Attr("input_type_6", AttrValue::kDataTypeList)
  .Attr("output_size", AttrValue::kInt)
  .Attr("statis_type", AttrValue::kString)
  .Attr("var_name", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("var_names", AttrValue::kString);

XDL_REGISTER_KERNEL(PsMergedSparseStatisOp, PsMergedSparseStatisOp).Device("CPU");

} // namespace xdl


