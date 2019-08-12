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

#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/lib/timer.h"
#include "xdl/core/lib/tbb_concurrent_queue.h"
#include "xdl/data_io/data_io.h"

namespace xdl {

template <typename T>
class GetBatchOp: public OpKernel {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    std::string ds;
    XDL_CHECK_STATUS(ctx->GetAttr("ds", &ds));
    XDL_CHECK_STATUS(ctx->GetAttr("sparse_count", &sparse_count_));
    XDL_CHECK_STATUS(ctx->GetAttr("dense_count", &dense_count_));
    XDL_CHECK_STATUS(ctx->GetAttr("indicator_count", &indicator_count_));
    XDL_CHECK_STATUS(ctx->GetAttr("tag_cnt", &tag_cnt_));

    data_io_ = io::DataIOMap::Instance()->Get(ds);
    XDL_CHECK(data_io_ != nullptr);

    unique_ids_ = data_io_->GetUniqueIds();
    sample_id_ = 0;
    return Status::Ok();
  }

  Status Compute(OpKernelContext* ctx) override {
    //XDL_TIMER_SCOPE(get_batch_timer);
    using TensorList = std::vector<Tensor>;

    auto batch = data_io_->GetBatch();
    if (data_io_->finished()) {
      XDL_LOG(DEBUG) << "game over";
      TBBConcurrentQueue::Global()->SetFinished();
      if (data_io_->IsStreaming()) {
        return Status::ReachEnd("reach end of current window");
      }
      return Status::OutOfRange("game over ");
    }

    XDL_CHECK(batch != nullptr);

    /// skey
    auto blk = batch->Get(io::kSKeyName);
    if (blk == nullptr) {
      ctx->SetOutput("skbuf", Tensor(ctx->GetDevice(), TensorShape({0}), types::kInt8));
    } else {
      XDL_DCHECK(blk->ts_[io::Block::kSBuf] != nullptr);
      ctx->SetOutput("skbuf", *blk->ts_[io::Block::kSBuf]);
    }

    /// label
    blk = batch->Get(io::kLabelName);
    XDL_CHECK(blk != nullptr && blk->ts_[io::Block::kValue] != nullptr);
    ctx->SetOutput("label", *blk->ts_[io::Block::kValue]);

    // tag
    Tensor tag;
    ctx->AllocateOutput("tag", TensorShape({}), &tag);
    *(tag.Raw<int32_t>()) = (tag_cnt_ == 0 ? 0 : sample_id_++ % tag_cnt_);
    ctx->SetOutput("tag", tag);

    /// feature
    TensorList out_indices;
    TensorList out_ids;
    TensorList out_segments;
    TensorList out_svalues;
    TensorList out_dvalues;
    TensorList out_indicators;
    TensorList out_sample_indices;
    TensorList out_sample_segments;

    auto out_list = data_io_->sparse_list();
    XDL_CHECK(out_list.size() == sparse_count_);
    for (auto &o : out_list) {
      blk = batch->Get(o);
      XDL_CHECK(blk != nullptr);

      if (unique_ids_) {
        XDL_DCHECK(blk->ts_[io::Block::kUKey] != nullptr && blk->ts_[io::Block::kUKey]->Type() == types::kInt64);
        XDL_DCHECK(blk->ts_[io::Block::kIndex] != nullptr && blk->ts_[io::Block::kIndex]->Type() == types::kInt32);
        XDL_DCHECK(blk->ts_[io::Block::kSIndex] != nullptr && blk->ts_[io::Block::kSIndex]->Type() == types::kInt32);
        XDL_DCHECK(blk->ts_[io::Block::kSSegment] != nullptr && blk->ts_[io::Block::kSSegment]->Type() == types::kInt32);
        out_ids.push_back(*blk->ts_[io::Block::kUKey]);
        out_indices.push_back(*blk->ts_[io::Block::kIndex]);
        out_sample_indices.push_back(*blk->ts_[io::Block::kSIndex]);
        out_sample_segments.push_back(*blk->ts_[io::Block::kSSegment]);
      } else {
        XDL_DCHECK(blk->ts_[io::Block::kKey] != nullptr && blk->ts_[io::Block::kKey]->Type() == types::kInt64);
        XDL_DCHECK(blk->ts_[io::Block::kIndex] == nullptr);
        XDL_DCHECK(blk->ts_[io::Block::kSIndex] == nullptr);
        XDL_DCHECK(blk->ts_[io::Block::kSSegment] == nullptr);
        out_ids.push_back(*blk->ts_[io::Block::kKey]);
        out_indices.push_back(Tensor(ctx->GetDevice(), TensorShape({0}), types::kInt32));
        out_sample_indices.push_back(Tensor(ctx->GetDevice(), TensorShape({0}), types::kInt32));
        out_sample_segments.push_back(Tensor(ctx->GetDevice(), TensorShape({0}), types::kInt32));
      }

      if (blk->ts_[io::Block::kValue] != nullptr) {
        XDL_DCHECK(blk->ts_[io::Block::kValue]->Type() == types::kFloat);
        out_svalues.push_back(*blk->ts_[io::Block::kValue]);
      } else {
        out_svalues.push_back(Tensor(ctx->GetDevice(), TensorShape({0}), types::kFloat));
      }

      XDL_DCHECK(blk->ts_[io::Block::kSegment] != nullptr && blk->ts_[io::Block::kSegment]->Type() == types::kInt32);
      out_segments.push_back(*blk->ts_[io::Block::kSegment]);
    }

    XDL_CHECK_STATUS(ctx->SetOutputList("indices", out_indices));
    XDL_CHECK_STATUS(ctx->SetOutputList("ids", out_ids));
    XDL_CHECK_STATUS(ctx->SetOutputList("segments", out_segments));
    XDL_CHECK_STATUS(ctx->SetOutputList("svalues", out_svalues));
    XDL_CHECK_STATUS(ctx->SetOutputList("sample_indices", out_sample_indices));
    XDL_CHECK_STATUS(ctx->SetOutputList("sample_segments", out_sample_segments));

    out_list = data_io_->dense_list();
    XDL_DCHECK(out_list.size() == dense_count_);
    for (auto &o : out_list) {
      blk = batch->Get(o);
      XDL_DCHECK(blk != nullptr);
      XDL_DCHECK(blk->ts_[io::Block::kValue] != nullptr && blk->ts_[io::Block::kValue]->Type() == types::kFloat);
      out_dvalues.push_back(*blk->ts_[io::Block::kValue]);
    }

    XDL_CHECK_STATUS(ctx->SetOutputList("dvalues", out_dvalues));

    /// indicator

    XDL_DCHECK(data_io_->ntable() == indicator_count_ + 1);
    for (int i = 0; i < indicator_count_; ++i) {
      blk = batch->Get(io::kIndicatorPrefix+std::to_string(i));
      XDL_DCHECK(blk != nullptr);
      XDL_DCHECK(blk->ts_[io::Block::kIndex] != nullptr);
      out_indicators.push_back(*blk->ts_[io::Block::kIndex]);
    }

    XDL_CHECK_STATUS(ctx->SetOutputList("indicators", out_indicators));

    return Status::Ok();
  }

 private:
  long sparse_count_;
  long dense_count_;
  long indicator_count_;
  bool unique_ids_;
  int64_t tag_cnt_;
  int64_t sample_id_;
  io::DataIO *data_io_;
};

XDL_DEFINE_OP(GetBatch)
  .Attr("ds", AttrValue::kString)
  .Attr("sparse_count", AttrValue::kInt)
  .Attr("dense_count", AttrValue::kInt)
  .Attr("indicator_count", AttrValue::kInt)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("tag_cnt", AttrValue::kInt, 0)
  .OutputList("indicators", DataType::kInt32, "indicator_count")
  .OutputList("indices", DataType::kInt32, "sparse_count")
  .OutputList("ids", DataType::kInt64, "sparse_count")
  .OutputList("segments", DataType::kInt32, "sparse_count")
  .OutputList("svalues", "dtype", "sparse_count")
  .OutputList("dvalues", "dtype", "dense_count")
  .OutputList("sample_indices", DataType::kInt32, "sparse_count")
  .OutputList("sample_segments", DataType::kInt32, "sparse_count")
  .Output("skbuf", DataType::kInt8)
  .Output("label", "dtype")
  .Output("tag", DataType::kInt32);

#define REGISTER_KERNEL(T)                     \
  XDL_REGISTER_KERNEL(GetBatch, GetBatchOp<T>) \
  .Device("CPU")                               \
  .AttrDataType<T>("dtype");

REGISTER_KERNEL(int32_t);
REGISTER_KERNEL(int64_t);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

}  // namespace xdl
