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

#include "gtest/gtest.h"
#include "ps-plus/common/data.h"
#include "ps-plus/server/udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/thread_pool.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include <random>
#include <mutex>

using ps::server::Udf;
using ps::server::UdfContext;
using ps::server::UdfRegistry;
using ps::server::Variable;
using ps::server::Slices;
using ps::initializer::ConstantInitializer;
using ps::Initializer;
using ps::DataType;
using ps::TensorShape;
using ps::Tensor;
using ps::Data;
using ps::WrapperData;
using ps::ThreadPool;
using ps::Status;
using std::cout;
using std::endl;
using std::vector;

TEST(AggregateSlice, AggregateSliceSparse) {
    UdfRegistry* udf_registry = UdfRegistry::Get("AggregateSlice");
    Udf* udf = udf_registry->Build(vector<size_t>({0, 1, 2, 3}), vector<size_t>({4, 5}));
    UdfContext* ctx = new UdfContext;
    Variable* var = new Variable(new Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(0.0), Tensor::TType::kSegment, true), nullptr, "var");
    ctx->SetVariableName("var");
    vector<Slices> slices(1, Slices{.slice_size = 8, .slice_id = vector<size_t>({3, 5}), .dim_part = 1, .variable = var, .writable = true});
    vector<Tensor> grad(1, Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(1.0)));
    ctx->SetData(0, new WrapperData<vector<Slices> >(slices), true);
    ctx->SetData(1, new WrapperData<int64_t>(1), true);
    ctx->SetData(2, new WrapperData<int32_t>(3), true);
    ctx->SetData(3, new WrapperData<vector<Tensor> >(grad), true);
    ctx->SetVariable(var);
    
    Data* output;
    EXPECT_TRUE(udf->Run(ctx).IsOk());
    EXPECT_TRUE(ctx->GetData(4, &output).IsOk());
    vector<Slices>& out_slices = dynamic_cast<WrapperData<vector<Slices> >*>(output)->Internal();
    EXPECT_EQ(1ul, out_slices.size());
    EXPECT_EQ(8ul, out_slices[0].slice_size);
    EXPECT_EQ(0ul, out_slices[0].slice_id.size());
    EXPECT_EQ(1, out_slices[0].dim_part);
    ASSERT_TRUE(ctx->GetData(5, &output).IsOk());
    vector<Tensor>& out_grad = dynamic_cast<WrapperData<vector<Tensor> >*>(output)->Internal();
    ASSERT_EQ(1ul, out_grad.size());
    EXPECT_TRUE(out_grad[0].Shape().IsScalar());

    vector<Slices> slices1(1, Slices{.slice_size = 8, .slice_id = std::vector<size_t>({4, 5}), .dim_part = 1, .variable = var, .writable = true});
    vector<Tensor> grad1(1, Tensor(DataType::kFloat, TensorShape({2, 8}), new ConstantInitializer(2.0)));
    ctx->SetData(0, new WrapperData<vector<Slices> >(slices1), true);
    ctx->SetData(1, new WrapperData<int64_t>(1), true);
    ctx->SetData(2, new WrapperData<int32_t>(3), true);
    ctx->SetData(3, new WrapperData<vector<Tensor> >(grad1), true);
    
    EXPECT_TRUE(udf->Run(ctx).IsOk());
    EXPECT_TRUE(ctx->GetData(4, &output).IsOk());
    vector<Slices>& out_slices1 = dynamic_cast<WrapperData<vector<Slices> >*>(output)->Internal();
    EXPECT_EQ(1ul, out_slices1.size());
    EXPECT_EQ(8ul, out_slices1[0].slice_size);
    EXPECT_EQ(0ul, out_slices1[0].slice_id.size());
    EXPECT_EQ(1, out_slices1[0].dim_part);
    EXPECT_TRUE(ctx->GetData(5, &output).IsOk());
    vector<Tensor>& out_grad1 = dynamic_cast<WrapperData<vector<Tensor> >*>(output)->Internal();
    EXPECT_EQ(1ul, out_grad1.size());
    EXPECT_TRUE(out_grad1[0].Shape().IsScalar());

    vector<Slices> slices2(1, Slices{.slice_size = 8, .slice_id = std::vector<size_t>({3}), .dim_part = 1, .variable = var, .writable = true});
    vector<Tensor> grad2(1, Tensor(DataType::kFloat, TensorShape({1, 8}), new ConstantInitializer(3.0)));
    ctx->SetData(0, new WrapperData<vector<Slices> >(slices2), true);
    ctx->SetData(1, new WrapperData<int64_t>(1), true);
    ctx->SetData(2, new WrapperData<int32_t>(3), true);
    ctx->SetData(3, new WrapperData<vector<Tensor> >(grad2), true);
    
    EXPECT_TRUE(udf->Run(ctx).IsOk());
    EXPECT_TRUE(ctx->GetData(4, &output).IsOk());
    vector<Slices>& out_slices2 = dynamic_cast<WrapperData<vector<Slices> >*>(output)->Internal();
    EXPECT_EQ(1, out_slices2.size());
    EXPECT_EQ(8ul, out_slices2[0].slice_size);
    ASSERT_EQ(3ul, out_slices2[0].slice_id.size());
    EXPECT_EQ(12, out_slices2[0].slice_id[0] + out_slices2[0].slice_id[1] + out_slices2[0].slice_id[2]);
    EXPECT_EQ(1, out_slices[0].dim_part);
    EXPECT_TRUE(ctx->GetData(5, &output).IsOk());
    vector<Tensor>& out_grad2 = dynamic_cast<WrapperData<vector<Tensor> >*>(output)->Internal();
    EXPECT_EQ(24ul, out_grad2[0].Shape().NumElements());
    for (size_t j = 0; j < 3; j++) {
      if (out_slices2[0].slice_id[j] == 3) {
        for (size_t i = 0; i < 8; i++) {
          EXPECT_FLOAT_EQ(4/3.0, *(out_grad2[0].Raw<float>(j) + i));
        }
      } else if (out_slices2[0].slice_id[j] == 4) {
        for (size_t i = 0; i < 8; i++) {    
          EXPECT_FLOAT_EQ(2/3.0, *(out_grad2[0].Raw<float>(j) + i));
        }
      } else {
        for (size_t i = 0; i < 8; i++) {        
          EXPECT_FLOAT_EQ(1, *(out_grad2[0].Raw<float>(j) + i));
        }
      }
    }

    vector<Slices> slices3(1, Slices{.slice_size = 8, .slice_id = std::vector<size_t>({}), .dim_part = 1, .variable = var, .writable = true});
    vector<Tensor> grad3(1, Tensor(DataType::kFloat, TensorShape(), new ConstantInitializer(3.0)));
    ctx->SetData(0, new WrapperData<vector<Slices> >(slices3), true);
    ctx->SetData(1, new WrapperData<int64_t>(1), true);
    ctx->SetData(2, new WrapperData<int32_t>(3), true);
    ctx->SetData(3, new WrapperData<vector<Tensor> >(grad3), true);
    EXPECT_TRUE(udf->Run(ctx).IsOk());
    delete ctx;
    delete var;
    delete udf;
}

TEST(AggregateSlice, AggregateSliceDense) {
    UdfRegistry* udf_registry = UdfRegistry::Get("AggregateSlice");
    Udf* udf = udf_registry->Build(std::vector<size_t>({0, 1, 2, 3}), std::vector<size_t>({4, 5}));
    UdfContext* ctx = new UdfContext;
    
    Variable* var = new Variable(new Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(0)), nullptr, "dense");
    ctx->SetVariable(var);
    ctx->SetVariableName("dense");
    vector<Slices> slices(1, Slices{.slice_size = 32, .slice_id = std::vector<size_t>({0}), .dim_part = -1, .variable = var, .writable = true});
    vector<Tensor> grad(1, Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(1.0)));
    ctx->SetData(0, new WrapperData<vector<Slices> >(slices), true);
    ctx->SetData(1, new WrapperData<int64_t>(1), true);
    ctx->SetData(2, new WrapperData<int32_t>(2), true);
    ctx->SetData(3, new WrapperData<vector<Tensor> >(grad), true);

    Data* output;
    EXPECT_TRUE(udf->Run(ctx).IsOk());
    EXPECT_TRUE(ctx->GetData(4, &output).IsOk());
    vector<Slices>& out_slices1 = dynamic_cast<WrapperData<vector<Slices> >*>(output)->Internal();
    EXPECT_EQ(1, out_slices1.size());
    EXPECT_EQ(32ul, out_slices1[0].slice_size);
    EXPECT_EQ(0ul, out_slices1[0].slice_id.size());
    EXPECT_EQ(-1, out_slices1[0].dim_part);
    EXPECT_TRUE(ctx->GetData(5, &output).IsOk());
    vector<Tensor>& out_grad1 = dynamic_cast<WrapperData<vector<Tensor> >*>(output)->Internal();
    EXPECT_EQ(1, out_grad1.size());
    EXPECT_TRUE(out_grad1[0].Shape().IsScalar());

    vector<Tensor> grad2(1, Tensor(DataType::kFloat, TensorShape({4, 8}), new ConstantInitializer(2.0)));
    ctx->SetData(0, new WrapperData<vector<Slices> >(slices), true);
    ctx->SetData(1, new WrapperData<int64_t>(1), true);
    ctx->SetData(2, new WrapperData<int32_t>(2), true);
    ctx->SetData(3, new WrapperData<vector<Tensor> >(grad2), true);
    
    EXPECT_TRUE(udf->Run(ctx).IsOk());
    EXPECT_TRUE(ctx->GetData(4, &output).IsOk());
    vector<Slices>& out_slices2 = dynamic_cast<WrapperData<vector<Slices> >*>(output)->Internal();
    EXPECT_EQ(1, out_slices2.size());
    EXPECT_EQ(32ul, out_slices2[0].slice_size);
    EXPECT_EQ(1ul, out_slices2[0].slice_id.size());
    EXPECT_EQ(0, out_slices2[0].slice_id[0]);  
    EXPECT_EQ(-1, out_slices2[0].dim_part);
    EXPECT_TRUE(ctx->GetData(5, &output).IsOk());
    vector<Tensor>& out_grad2 = dynamic_cast<WrapperData<vector<Tensor> >*>(output)->Internal();
    EXPECT_EQ(1, out_grad2.size());
    EXPECT_EQ(32ul, out_grad2[0].Shape().NumElements());
    
    for (size_t i = 0; i < 32; i++) {
        EXPECT_FLOAT_EQ(1.5, out_grad2[0].Raw<float>()[i]);
    }
    delete var;
    delete ctx;
    delete udf;
    
}

/*
TEST(AggregateSlice, AggregateSliceSparseBenchMark) {
    ThreadPool* queue_ = new ThreadPool(8);
    UdfRegistry* udf_registry = UdfRegistry::Get("AggregateSlice");
    Udf* udf = udf_registry->Build(std::vector<size_t>({0, 1, 2, 3}), std::vector<size_t>({0, 1}));
    Variable* var = new Variable(new Tensor(DataType::kFloat, TensorShape({500000, 8}), new ConstantInitializer(0.0), true), nullptr);
    double total_time = 0;
    std::mutex lock_;
    clock_t real_start = clock();
    for (int32_t i = 0; i < 20; i++) {
        queue_->Schedule([&] {
                                  clock_t start,end;
                                  std::vector<size_t> ids(100000);
                                  std::default_random_engine e;
                                  for (size_t i = 0; i < 100000; i++) {
                                      ids[i] = e();
                                  }
                                  Slices slices{.slice_size = 8, .slice_id = ids, .dim_part = 1, .variable = var, .writable = true};
                                  UdfContext* ctx = new UdfContext;
                                  ctx->SetData(0, new WrapperData<Slices>(slices), true);
                                  ctx->SetData(1, new WrapperData<int64_t>(0), true);
                                  ctx->SetData(2, new WrapperData<int32_t>(20), true);
                                  ctx->SetData(3, new WrapperData<Tensor>(DataType::kFloat, TensorShape({100000, 8}), new ConstantInitializer(1.0), true), true);
                                  ctx->SetVariable(var);
                                  start=clock();
                                  Status status = udf->Run(ctx);
                                  end=clock();
                                  EXPECT_TRUE(status.IsOk());
                                  Data* output;
                                  EXPECT_TRUE(ctx->GetData(0, &output).IsOk());
                                  delete ctx;
                                  double one_time=(double)(end-start);
                                  lock_.lock();
                                  total_time += one_time;
                                  lock_.unlock();
                              });
    }
    delete queue_;
    clock_t real_end = clock();    
    cout << "Total time: " << total_time << endl;
    cout << "Real  time: " << real_end - real_start << endl;    
    delete var;
    delete udf;
}
*/
