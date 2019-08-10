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
#include "xdl/core/lib/singleton.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "ps-plus/common/hdfs_data_source.h"
#include <sys/time.h>

namespace xdl {

class HdfsDataSourceManager : public Singleton<HdfsDataSourceManager> {
 public:
  static constexpr int kDataSourceSize = 1024;
  std::string Spec(std::string filename, int server_size, int rank, int value_length) {
    return filename + "?server_size=" + std::to_string(server_size) +
      "&rank=" + std::to_string(rank) + "&value_length=" + std::to_string(value_length);
  }

  Status GetDataSource(std::string filename, int server_size, int rank, int value_length, ps::HdfsDataSource** result) {
    std::unique_lock<std::mutex> lock(mu_);
    std::string spec = Spec(filename, server_size, rank, value_length);
    auto iter = map_.find(spec);
    if (iter != map_.end()) {
      *result = iter->second.get();
      return Status::Ok();
    }
    std::unique_ptr<ps::HdfsDataSource> data_source(new ps::HdfsDataSource(filename, kDataSourceSize));
    XDL_CHECK_STATUS(PS2XDL::ConvertStatus(data_source->Init(rank, server_size, value_length)));
    *result = data_source.get();
    map_[spec] = std::move(data_source);
    return Status::Ok();
  }
 private:
  std::mutex mu_;
  std::unordered_map<std::string, std::unique_ptr<ps::HdfsDataSource>> map_;
};

class HdfsDataSourceOp : public OpKernel {
  Status Init(OpKernelConstruction* ctx) override {
    std::string filename;
    int64_t server_size;
    int64_t rank;
    XDL_CHECK_STATUS(ctx->GetAttr("filename", &filename));
    XDL_CHECK_STATUS(ctx->GetAttr("server_size", &server_size));
    XDL_CHECK_STATUS(ctx->GetAttr("rank", &rank));
    XDL_CHECK_STATUS(ctx->GetAttr("size", &size_));
    XDL_CHECK_STATUS(ctx->GetAttr("dtype", &dtype_));
    value_length_ = size_ * SizeOfType(dtype_);
    XDL_CHECK_STATUS(HdfsDataSourceManager::Instance()->GetDataSource(filename, server_size, rank, value_length_, &data_source_));
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor input, rst;
    XDL_CHECK_STATUS(ctx->GetInput(0, &input));
    XDL_CHECK_COND(input.Shape().Size() == 1,
                   Status::ArgumentError("input should be 1-rank"));
    int64_t len = input.Shape()[0];
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape({len, size_}), &rst));
    std::vector<int64_t> inputx(input.Raw<int64_t>(), input.Raw<int64_t>() + len);
    std::vector<ps::DataClosure> data_closures;
    XDL_CHECK_STATUS(PS2XDL::ConvertStatus(data_source_->BatchGet(inputx, &data_closures)));
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < data_closures.size(); i++) {
      memcpy(rst.Raw<char>() + value_length_ * i, data_closures[i].data, value_length_);
    }
    return Status::Ok();
  }
 private:
  int64_t size_;
  DataType dtype_;
  size_t value_length_;
  ps::HdfsDataSource* data_source_;
};

XDL_DEFINE_OP(HdfsDataSourceOp)
  .Attr("filename", AttrValue::kString)
  .Attr("server_size", AttrValue::kInt)
  .Attr("rank", AttrValue::kInt)
  .Attr("size", AttrValue::kInt)
  .Attr("dtype", AttrValue::kDataType)
  .Input("input", DataType::kInt64)
  .Output("rst", "dtype");

XDL_REGISTER_KERNEL(HdfsDataSourceOp, HdfsDataSourceOp)
  .Device("CPU");

class HdfsDataSourceOpV2 : public OpKernel {
  Status Init(OpKernelConstruction* ctx) override {
    std::string filename;
    int64_t server_size;
    int64_t rank;
    XDL_CHECK_STATUS(ctx->GetAttr("filename", &filename));
    XDL_CHECK_STATUS(ctx->GetAttr("server_size", &server_size));
    XDL_CHECK_STATUS(ctx->GetAttr("rank", &rank));
    XDL_CHECK_STATUS(ctx->GetAttr("size", &size_));
    XDL_CHECK_STATUS(ctx->GetAttr("dtype", &dtype_));
    value_length_ = size_ * SizeOfType(dtype_);
    XDL_CHECK_STATUS(HdfsDataSourceManager::Instance()->GetDataSource(filename, server_size, rank, value_length_, &data_source_));
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    Tensor input, rst, ids_rst;
    XDL_CHECK_STATUS(ctx->GetInput(0, &input));
    XDL_CHECK_COND(input.Shape().Size() == 1,
                   Status::ArgumentError("input should be 1-rank"));
    int64_t len = input.Shape()[0];
    std::vector<int64_t> inputx(input.Raw<int64_t>(), input.Raw<int64_t>() + len);
    std::vector<ps::DataClosure> data_closures;
    std::vector<int64_t> ids;
    data_source_->BatchGetV2(inputx, &data_closures, &ids);
    XDL_CHECK_STATUS(ctx->AllocateOutput(0, TensorShape({data_closures.size(), size_}), &rst));
    XDL_CHECK_STATUS(ctx->AllocateOutput(1, TensorShape({ids.size()}), &ids_rst));
    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < data_closures.size(); i++) {
      memcpy(rst.Raw<char>() + value_length_ * i, data_closures[i].data, value_length_);
    }
    memcpy(ids_rst.Raw<char>(), &ids[0], sizeof(int64_t) * ids.size());
    return Status::Ok();
  }
 private:
  int64_t size_;
  DataType dtype_;
  size_t value_length_;
  ps::HdfsDataSource* data_source_;
};

XDL_DEFINE_OP(HdfsDataSourceOpV2)
  .Attr("filename", AttrValue::kString)
  .Attr("server_size", AttrValue::kInt)
  .Attr("rank", AttrValue::kInt)
  .Attr("size", AttrValue::kInt)
  .Attr("dtype", AttrValue::kDataType)
  .Input("input", DataType::kInt64)
  .Output("rst", "dtype")
  .Output("ids_rst", DataType::kInt64);

XDL_REGISTER_KERNEL(HdfsDataSourceOpV2, HdfsDataSourceOpV2)
  .Device("CPU");

}
