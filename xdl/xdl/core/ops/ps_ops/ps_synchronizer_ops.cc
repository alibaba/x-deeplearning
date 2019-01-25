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

namespace xdl {

inline int IntArg(OpKernelContext* ctx, int k, OpKernelBase::Callback& done) {
  Tensor value;
  xdl::Status status = ctx->GetInput(k, &value);
  if (!status.IsOk()) {
      done(status);
  }
  return value.Scalar<int>();
}

class PsAsynchronizeEnterOp: public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }
    
  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    int id = IntArg(ctx, 0, done);
    int staleness = IntArg(ctx, 1, done);
    int worker_count = IntArg(ctx, 2, done);
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };
    client->AsynchronizeEnter(id, staleness, worker_count, cb);
  }
};

class PsSynchronizeEnterOp: public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }
    
  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    int id = IntArg(ctx, 0, done);
    int worker_count = IntArg(ctx, 1, done);
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };
    client->SynchronizeEnter(id, worker_count, cb);
  }
};

class PsSynchronizeLeaveOp: public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);    
    int id = IntArg(ctx, 0, done);
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };
    client->SynchronizeLeave(id, cb);
  }
};

class WorkerReportFinishOp: public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);    
    int id = IntArg(ctx, 0, done);
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };    
    client->WorkerReportFinish(id, cb);
  }
};

class WorkerBarrierOp: public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }
  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);    
    int id = IntArg(ctx, 0, done);
    int worker_count = IntArg(ctx, 1, done);
    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };    
    client->WorkerBarrier(id, worker_count, cb);
  }
};

XDL_DEFINE_OP(PsAsynchronizeEnterOp)
  .Input("id", DataType::kInt32)
  .Input("staleness", DataType::kInt32)
  .Input("worker_count", DataType::kInt32);

XDL_DEFINE_OP(PsSynchronizeEnterOp)
  .Input("id", DataType::kInt32)
  .Input("worker_count", DataType::kInt32);

XDL_DEFINE_OP(PsSynchronizeLeaveOp)
  .Input("id", DataType::kInt32);

XDL_DEFINE_OP(WorkerReportFinishOp)
  .Input("id", DataType::kInt32);

XDL_DEFINE_OP(WorkerBarrierOp)
  .Input("id", DataType::kInt32)
  .Input("worker_count", DataType::kInt32);

XDL_REGISTER_KERNEL(PsAsynchronizeEnterOp, PsAsynchronizeEnterOp).Device("CPU");
XDL_REGISTER_KERNEL(PsSynchronizeEnterOp, PsSynchronizeEnterOp).Device("CPU");
XDL_REGISTER_KERNEL(PsSynchronizeLeaveOp, PsSynchronizeLeaveOp).Device("CPU");
XDL_REGISTER_KERNEL(WorkerReportFinishOp, WorkerReportFinishOp).Device("CPU");
XDL_REGISTER_KERNEL(WorkerBarrierOp, WorkerBarrierOp).Device("CPU");

}

