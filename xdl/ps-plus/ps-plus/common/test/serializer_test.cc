#include <cstring>

#include "gtest/gtest.h"

#define private public

#include "ps-plus/common/data.h"
#include "ps-plus/common/tensor.h"
#include "ps-plus/common/tensor_shape.h"
#include "ps-plus/common/serializer.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/initializer/none_initializer.h"
#include "ps-plus/common/initializer/truncated_normal_initializer.h"
#include "ps-plus/common/initializer/uniform_unit_scaling_initializer.h"
#include "ps-plus/common/initializer/variance_scaling_initializer.h"
#include "ps-plus/message/variable_info.h"
#include "ps-plus/message/cluster_info.h"
#include "ps-plus/message/udf_chain_register.h"

#undef private

using ps::serializer::Serializer;
using ps::serializer::Deserializer;
using ps::serializer::SerializeAny;
using ps::serializer::DeserializeAny;
using ps::serializer::Fragment;
using ps::serializer::MemGuard;
using ps::Status;
using ps::Status;
using ps::DataType;
using ps::Data;
using ps::WrapperData;
using std::string;
using ps::UdfChainRegister;
using ps::server::Slices;

void FragmentConcat(const std::vector<Fragment>& bufs, Fragment* merge) {
  size_t total_size = 0;
  for (auto& item: bufs) {
    total_size += item.size;
  }

  merge->base = new char[total_size];
  size_t offset = 0;
  for (auto& item: bufs) {
    std::memcpy(merge->base + offset, item.base, item.size);
    offset += item.size;
  }
  merge->size = total_size;
}

TEST(SerializerTest, Serializer) {
  {
    MemGuard mem_guard;
    WrapperData<bool>* data = new WrapperData<bool>(true);
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());
    ps::Data* result = nullptr;
    size_t len;
    Status s = DeserializeAny<Data>(id, &bufs[0], 0, &result, &len, mem_guard);
    EXPECT_TRUE(s.IsOk());
    WrapperData<bool>* r = dynamic_cast<WrapperData<bool>*>(result);    
    EXPECT_EQ(true, r->Internal());
    EXPECT_EQ(sizeof(bool), len);
    delete data;
    delete result;
  }

  {
    MemGuard mem_guard;
    WrapperData<string>* data = new WrapperData<string>("test");
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());
    
    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<string>* r = dynamic_cast<WrapperData<string>*>(result);    
    EXPECT_EQ(string("test"), r->Internal());
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    MemGuard mem_guard;
    WrapperData<ps::Status>* data = new WrapperData<ps::Status>(
        ps::Status::kArgumentError, "kArgumentError");
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<ps::Status>* r = dynamic_cast<WrapperData<ps::Status>*>(result);    
    EXPECT_EQ(1, r->Internal().Code());
    EXPECT_EQ(string("kArgumentError"), r->Internal().Msg());
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    MemGuard mem_guard;
    DataType type = DataType::kInt16;
    WrapperData<DataType>* data = new WrapperData<DataType>(type);
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Status s = DeserializeAny<Data>(id, &bufs[0], 0, &result, &len, mem_guard);
    EXPECT_TRUE(s.IsOk());
    WrapperData<DataType>* r = dynamic_cast<WrapperData<DataType>*>(result);    
    EXPECT_EQ(DataType::kInt16, r->Internal());
    delete data;
    delete result;
  }

  {
    MemGuard mem_guard;
    ps::TensorShape ts({2,8,2});
    WrapperData<ps::TensorShape>* data = new WrapperData<ps::TensorShape>(ts);
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    ps::Status st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    EXPECT_TRUE(st.IsOk());
    WrapperData<ps::TensorShape>* r = dynamic_cast<WrapperData<ps::TensorShape>*>(result);    
    EXPECT_EQ(ps::TensorShape({2,8,2}), r->Internal());
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    using Initializer = ps::Initializer;
    using ConstantInitializer = ps::initializer::ConstantInitializer;
    MemGuard mem_guard;
    Initializer* i = new ConstantInitializer(5.0);
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<ps::Initializer>(i, &id, &bufs, mem_guard).IsOk());

    ps::Initializer* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    ps::Status st = DeserializeAny<Initializer>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    ConstantInitializer* r = dynamic_cast<ConstantInitializer*>(result);
    EXPECT_TRUE(r != nullptr);
    int32_t buf[10];
    r->Init(buf, DataType::kInt32, 10);
    for (size_t i = 0; i < 10; ++i) {
      EXPECT_EQ(5, buf[i]);
    }

    delete i;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    using Initializer = ps::Initializer;
    using ConstantInitializer = ps::initializer::ConstantInitializer;
    MemGuard mem_guard;
    std::unique_ptr<Initializer> raw(new ConstantInitializer(5.0));
    WrapperData<std::unique_ptr<Initializer> >* data = 
      new WrapperData<std::unique_ptr<Initializer> >(std::move(raw));
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    ps::Status st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    WrapperData<std::unique_ptr<Initializer> >* r = 
      dynamic_cast<WrapperData<std::unique_ptr<Initializer> >*>(result);    
    Initializer* ci = r->Internal().get();
    EXPECT_TRUE(ci != nullptr);
    EXPECT_TRUE(dynamic_cast<ConstantInitializer*>(ci) != nullptr);
    int32_t buf[10];
    ci->Init(buf, DataType::kInt32, 10);
    for (size_t i = 0; i < 10; ++i) {
      EXPECT_EQ(5, buf[i]);
    }

    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    using Initializer = ps::Initializer;
    using NoneInitializer = ps::initializer::NoneInitializer;
    MemGuard mem_guard;
    std::unique_ptr<Initializer> raw(new NoneInitializer());
    WrapperData<std::unique_ptr<Initializer> >* data = 
      new WrapperData<std::unique_ptr<Initializer> >(std::move(raw));
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    ps::Status st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    WrapperData<std::unique_ptr<Initializer> >* r = 
      dynamic_cast<WrapperData<std::unique_ptr<Initializer> >*>(result);    
    Initializer* ci = r->Internal().get();
    EXPECT_TRUE(ci != nullptr);
    EXPECT_TRUE(dynamic_cast<NoneInitializer*>(ci) != nullptr);
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    using Initializer = ps::Initializer;
    using TruncatedNormalInitializer = ps::initializer::TruncatedNormalInitializer;
    MemGuard mem_guard;
    std::unique_ptr<Initializer> raw(new TruncatedNormalInitializer(-1, 1, 0));
    WrapperData<std::unique_ptr<Initializer> >* data = 
      new WrapperData<std::unique_ptr<Initializer> >(std::move(raw));
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    ps::Status st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    WrapperData<std::unique_ptr<Initializer> >* r = 
      dynamic_cast<WrapperData<std::unique_ptr<Initializer> >*>(result);    
    Initializer* i = r->Internal().get();
    EXPECT_TRUE(i != nullptr);
    EXPECT_TRUE(dynamic_cast<TruncatedNormalInitializer*>(i) != nullptr);
    TruncatedNormalInitializer* ti = dynamic_cast<TruncatedNormalInitializer*>(i);
    EXPECT_EQ(-1, ti->seed_);
    EXPECT_EQ(1.0, ti->mean_);
    EXPECT_EQ(0, ti->stddev_);
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    using Initializer = ps::Initializer;
    using UniformUnitScalingInitializer = ps::initializer::UniformUnitScalingInitializer;
    MemGuard mem_guard;
    std::unique_ptr<Initializer> raw(new UniformUnitScalingInitializer(ps::TensorShape({2,8}), -1, 1.0));
    WrapperData<std::unique_ptr<Initializer> >* data = 
      new WrapperData<std::unique_ptr<Initializer> >(std::move(raw));
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    ps::Status st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    WrapperData<std::unique_ptr<Initializer> >* r = 
      dynamic_cast<WrapperData<std::unique_ptr<Initializer> >*>(result);    
    Initializer* i = r->Internal().get();
    EXPECT_TRUE(i != nullptr);
    EXPECT_TRUE(dynamic_cast<UniformUnitScalingInitializer*>(i) != nullptr);
    UniformUnitScalingInitializer* ui = dynamic_cast<UniformUnitScalingInitializer*>(i);
    EXPECT_EQ(-1, ui->seed_);
    EXPECT_EQ(1.0, ui->factor_);
    EXPECT_EQ(ps::TensorShape({2,8}), ui->shape_);
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    using Initializer = ps::Initializer;
    using VarianceScalingInitializer = ps::initializer::VarianceScalingInitializer;
    MemGuard mem_guard;
    std::unique_ptr<Initializer> raw(new VarianceScalingInitializer(ps::TensorShape({2,8}), -1, 1.0, "fan_in", "normal"));
    WrapperData<std::unique_ptr<Initializer> >* data = 
      new WrapperData<std::unique_ptr<Initializer> >(std::move(raw));
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    ps::Status st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    WrapperData<std::unique_ptr<Initializer> >* r = 
      dynamic_cast<WrapperData<std::unique_ptr<Initializer> >*>(result);    
    Initializer* i = r->Internal().get();
    EXPECT_TRUE(i != nullptr);
    EXPECT_TRUE(dynamic_cast<VarianceScalingInitializer*>(i) != nullptr);
    VarianceScalingInitializer* vi = dynamic_cast<VarianceScalingInitializer*>(i);
    EXPECT_EQ(-1, vi->seed_);
    EXPECT_EQ(1.0, vi->scale_);
    EXPECT_EQ(ps::TensorShape({2,8}), vi->full_shape_);
    EXPECT_EQ(std::string("fan_in"), vi->mode_);
    EXPECT_EQ(std::string("normal"), vi->distribution_);
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    MemGuard mem_guard;
    ps::Tensor t(DataType::kInt32, ps::TensorShape({2,8}), new ps::initializer::ConstantInitializer(4));
    WrapperData<ps::Tensor>* data = new WrapperData<ps::Tensor>(std::move(t));
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<ps::Tensor>* r = dynamic_cast<WrapperData<ps::Tensor>*>(result);
    EXPECT_EQ(DataType::kInt32, r->Internal().Type());
    EXPECT_EQ(ps::TensorShape({2,8}), r->Internal().Shape());
    for (size_t i = 0; i < 16; ++i) {
      EXPECT_EQ(4, *(r->Internal().Raw<int32_t>() + i));
    }

    ps::Initializer* iz = r->Internal().state_->initializer.get();
    EXPECT_TRUE(dynamic_cast<ps::initializer::ConstantInitializer*>(iz) != nullptr);
    int32_t buf[4];
    iz->Init(buf, DataType::kInt32, 4);
    for (size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(4, buf[i]);
    }

    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    MemGuard mem_guard;
    ps::Tensor t(DataType::kFloat, 
                 ps::TensorShape({2,8}), 
                 new ps::initializer::TruncatedNormalInitializer(-1, 1, 0));
    WrapperData<ps::Tensor>* data = new WrapperData<ps::Tensor>(std::move(t));
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<ps::Tensor>* r = dynamic_cast<WrapperData<ps::Tensor>*>(result);
    EXPECT_EQ(DataType::kFloat, r->Internal().Type());
    EXPECT_EQ(ps::TensorShape({2,8}), r->Internal().Shape());
    for (size_t i = 0; i < 16; ++i) {
      EXPECT_EQ(1.0, *(r->Internal().Raw<float>() + i));
    }

    ps::Initializer* iz = r->Internal().state_->initializer.get();
    EXPECT_TRUE(dynamic_cast<ps::initializer::TruncatedNormalInitializer*>(iz) != nullptr);
    float buf[16];
    iz->Init(buf, DataType::kFloat, 16);
    for (size_t i = 0; i < 16; ++i) {
      EXPECT_EQ(1.0, buf[i]);
    }

    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    //vector<Slices> to vector<Tensor>
    MemGuard mem_guard;
    Slices slice;
    slice.slice_size = 2;
    slice.dim_part = 1;
    slice.slice_id.push_back(0);
    slice.slice_id.push_back(2);
    slice.slice_id.push_back(3);
    int32_t buf[] = {1, 2, 3, 4, 5, 6, 7, 8};
    ps::Tensor t(DataType::kInt32, ps::TensorShape({4,2}), nullptr, false, ps::Tensor::DEFAULT_SEGMENT_SIZE);
    for (size_t i = 0; i < 4; i++) {
      int32_t* p = t.Raw<int32_t>(i);
      p[0] = buf[i*2];
      p[1] = buf[i*2+1];
    }
    slice.variable = new ps::server::Variable(&t, nullptr, "");
    
    WrapperData<std::vector<ps::server::Slices> >* data = new WrapperData<std::vector<ps::server::Slices> >(std::vector<ps::server::Slices>{slice});
    size_t id;
    std::vector<Fragment> bufs;
    Status st = SerializeAny<Data>(data, &id, &bufs, mem_guard);
    EXPECT_TRUE(st.IsOk());

    ps::Data* result = nullptr;
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    EXPECT_TRUE(st.IsOk());
    WrapperData<std::vector<ps::Tensor> >* r = dynamic_cast<WrapperData<std::vector<ps::Tensor> >*>(result);
    EXPECT_EQ(1, r->Internal().size());
    ps::Tensor& rt = r->Internal()[0];
    EXPECT_EQ(DataType::kInt32, rt.Type());
    EXPECT_EQ(ps::TensorShape({3,2}), rt.Shape());
    int32_t expected[] = {1, 2, 5, 6, 7, 8};
    for (size_t i = 0; i < 6; ++i) {
      EXPECT_EQ(expected[i], *(rt.Raw<int32_t>() + i));
    }  
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }

  {
    //TensorSlices to Tensor
    MemGuard mem_guard;
    ps::server::TensorSlices slice;
    slice.slice_size = 2;
    slice.dim_part = 1;
    slice.slice_id.push_back(0);
    slice.slice_id.push_back(2);
    slice.slice_id.push_back(3);
    int32_t buf[] = {1, 2, 3, 4, 5, 6, 7, 8};
    ps::Tensor t(DataType::kInt32, ps::TensorShape({4,2}), nullptr, false, ps::Tensor::DEFAULT_SEGMENT_SIZE);
    for (size_t i = 0; i < 4; i++) {
      int32_t* p = t.Raw<int32_t>(i);
      p[0] = buf[i*2];
      p[1] = buf[i*2+1];      
    }
    slice.tensor = t;
    WrapperData<ps::server::TensorSlices>* data = new WrapperData<ps::server::TensorSlices>(slice);
    size_t id;
    std::vector<Fragment> bufs;
    Status st = SerializeAny<Data>(data, &id, &bufs, mem_guard);
    EXPECT_TRUE(st.IsOk());

    ps::Data* result = nullptr;
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    EXPECT_TRUE(st.IsOk());
    WrapperData<ps::Tensor>* r = dynamic_cast<WrapperData<ps::Tensor>*>(result);
    EXPECT_EQ(DataType::kInt32, r->Internal().Type());
    EXPECT_EQ(ps::TensorShape({3,2}), r->Internal().Shape());
    int32_t expected[] = {1, 2, 5, 6, 7, 8};
    for (size_t i = 0; i < 6; ++i) {
      EXPECT_EQ(expected[i], *(r->Internal().Raw<int32_t>() + i));
    }  
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }  

  {
    //vector<TensorSlices> to vector<Tensor>
    MemGuard mem_guard;
    ps::server::TensorSlices slice;
    slice.slice_size = 2;
    slice.dim_part = 1;
    slice.slice_id.push_back(0);
    slice.slice_id.push_back(2);
    slice.slice_id.push_back(3);
    int32_t buf[] = {1, 2, 3, 4, 5, 6, 7, 8};
    ps::Tensor t(DataType::kInt32, ps::TensorShape({4,2}), nullptr, false, ps::Tensor::DEFAULT_SEGMENT_SIZE);
    for (size_t i = 0; i < 4; i++) {
      int32_t* p = t.Raw<int32_t>(i);
      p[0] = buf[i*2];
      p[1] = buf[i*2+1];
    }
    slice.tensor = t;

    ps::server::TensorSlices slice2;
    slice2.slice_size = 2;
    slice2.dim_part = 1;
    slice2.slice_id.push_back(0);
    slice2.slice_id.push_back(3);
    int32_t buf2[] = {9, 10, 11, 12, 13, 14, 15, 16};
    ps::Tensor t2(DataType::kInt32, ps::TensorShape({4,2}), nullptr, false, ps::Tensor::DEFAULT_SEGMENT_SIZE);
    for (size_t i = 0; i < 4; i++) {
      int32_t* p = t2.Raw<int32_t>(i);
      p[0] = buf2[i*2];
      p[1] = buf2[i*2+1];
    }
    slice2.tensor = t2;
    WrapperData<std::vector<ps::server::TensorSlices> >* data = new WrapperData<std::vector<ps::server::TensorSlices> >(std::vector<ps::server::TensorSlices>{slice, slice2});
    size_t id;
    std::vector<Fragment> bufs;
    Status st = SerializeAny<Data>(data, &id, &bufs, mem_guard);
    EXPECT_TRUE(st.IsOk());

    ps::Data* result = nullptr;
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    st = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    EXPECT_TRUE(st.IsOk());
    WrapperData<std::vector<ps::Tensor> >* r = dynamic_cast<WrapperData<std::vector<ps::Tensor> >*>(result);
    EXPECT_EQ(2, r->Internal().size());
    ps::Tensor& rt = r->Internal()[0];
    EXPECT_EQ(DataType::kInt32, rt.Type());
    EXPECT_EQ(ps::TensorShape({3,2}), rt.Shape());
    int32_t expected[] = {1, 2, 5, 6, 7, 8};
    for (size_t i = 0; i < 6; ++i) {
      EXPECT_EQ(expected[i], *(rt.Raw<int32_t>() + i));
    }

    ps::Tensor& rt2 = r->Internal()[1];
    EXPECT_EQ(DataType::kInt32, rt2.Type());
    EXPECT_EQ(ps::TensorShape({2,2}), rt2.Shape());
    int32_t expected2[] = {9, 10, 15, 16};
    for (size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(expected2[i], *(rt2.Raw<int32_t>() + i));
    }      
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }
}

TEST(MessageSerializerTest, ServerInfoTest) {
  {
    MemGuard mem_guard;
    WrapperData<ps::ServerInfo>* data = new WrapperData<ps::ServerInfo>(
        42, 0, 1, "127.0.0.1", 80);
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<ps::ServerInfo>* si = dynamic_cast<WrapperData<ps::ServerInfo>*>(result);    
    EXPECT_EQ(42, si->Internal().GetServerType());
    EXPECT_EQ((ps::ServerId)0, si->Internal().GetId());
    EXPECT_EQ((ps::Version)1, si->Internal().GetVersion());
    EXPECT_EQ(string("127.0.0.1"), si->Internal().GetIp());
    EXPECT_EQ((uint16_t)80, si->Internal().GetPort());
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }
}

TEST(MessageSerializerTest, ClusterInfoTest) {
  {
    MemGuard mem_guard;
    WrapperData<ps::ClusterInfo>* data = new WrapperData<ps::ClusterInfo>();
    data->Internal().AddServer(ps::ServerInfo(42, 0, 0, "127.0.0.1", 80));
    data->Internal().AddServer(ps::ServerInfo(42, 1, 1, "127.0.0.1", 81));
    data->Internal().AddServer(ps::ServerInfo(42, 2, 2, "127.0.0.1", 82));
    data->Internal().AddServer(ps::ServerInfo(42, 3, 3, "127.0.0.1", 83));

    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<ps::ClusterInfo>* ci = dynamic_cast<WrapperData<ps::ClusterInfo>*>(result);    
    EXPECT_EQ((size_t)4, ci->Internal().GetServers().size());
    std::vector<ps::ServerInfo> expected_list;
    expected_list.emplace_back(ps::ServerInfo(42, 0, 0, "127.0.0.1", 80));
    expected_list.emplace_back(ps::ServerInfo(42, 1, 1, "127.0.0.1", 81));
    expected_list.emplace_back(ps::ServerInfo(42, 2, 2, "127.0.0.1", 82));
    expected_list.emplace_back(ps::ServerInfo(42, 3, 3, "127.0.0.1", 83));
    size_t i = 0;
    for (auto& it: ci->Internal().GetServers()) {
      EXPECT_TRUE(it == expected_list[i]);
      ++i;
    }

    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }
}

TEST(MessageSerializerTest, VariableInfoTest) {
  {
    MemGuard mem_guard;
    ps::VariableInfo raw;
    raw.type = ps::VariableInfo::kIndex;
    raw.name = "test_var";
    raw.parts.emplace_back(ps::VariableInfo::Part({0, 1}));
    raw.parts.emplace_back(ps::VariableInfo::Part({2, 3}));
    raw.shape = {2, 8};
    raw.datatype = ps::types::kInt16;
    raw.args.insert({"name", "test_name"});
    raw.args.insert({"value", "test_value"});
    ps::WrapperData<ps::VariableInfo>* data = new ps::WrapperData<ps::VariableInfo>(raw);

    size_t id;
    std::vector<Fragment> bufs;
    ps::Status st = SerializeAny<Data>(data, &id, &bufs, mem_guard);
    EXPECT_TRUE(st.IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<ps::VariableInfo>* vi = dynamic_cast<WrapperData<ps::VariableInfo>*>(result);    
    EXPECT_EQ(ps::VariableInfo::kIndex, vi->Internal().type);
    EXPECT_EQ(string("test_var"), vi->Internal().name);
    EXPECT_EQ(ps::types::kInt16, vi->Internal().datatype);
    EXPECT_EQ(std::vector<int64_t>({2, 8}), vi->Internal().shape);    
    EXPECT_EQ((size_t)0, vi->Internal().parts[0].server);
    EXPECT_EQ((size_t)1, vi->Internal().parts[0].size);
    EXPECT_EQ((size_t)2, vi->Internal().parts[1].server);
    EXPECT_EQ((size_t)3, vi->Internal().parts[1].size);
    EXPECT_EQ(raw.args, vi->Internal().args);        
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }
}

TEST(MessageSerializerTest, VariableInfoCollectionTest) {
  {
    MemGuard mem_guard;
    ps::VariableInfo raw;
    raw.type = ps::VariableInfo::kIndex;
    raw.name = "test_var";
    raw.parts.emplace_back(ps::VariableInfo::Part({0, 1}));
    raw.parts.emplace_back(ps::VariableInfo::Part({2, 3}));
    raw.shape = {2, 8};
    raw.datatype = ps::types::kInt16;
    raw.args.insert({"name", "test_name"});
    raw.args.insert({"value", "test_value"});

    ps::VariableInfoCollection raw_vic;
    raw_vic.infos.emplace_back(raw);
    
    ps::WrapperData<ps::VariableInfoCollection>* data = 
      new ps::WrapperData<ps::VariableInfoCollection>(raw_vic);

    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<ps::VariableInfoCollection>* vic = 
      dynamic_cast<WrapperData<ps::VariableInfoCollection>*>(result);    
    EXPECT_EQ((size_t)1, vic->Internal().infos.size());    
    ps::VariableInfo& vi = vic->Internal().infos[0];
    EXPECT_EQ(ps::VariableInfo::kIndex, vi.type);
    EXPECT_EQ(string("test_var"), vi.name);
    EXPECT_EQ(ps::types::kInt16, vi.datatype);
    EXPECT_EQ(std::vector<int64_t>({2, 8}), vi.shape);    
    EXPECT_EQ((size_t)0, vi.parts[0].server);
    EXPECT_EQ((size_t)1, vi.parts[0].size);
    EXPECT_EQ((size_t)2, vi.parts[1].server);
    EXPECT_EQ((size_t)3, vi.parts[1].size);
    EXPECT_EQ(raw.args, vi.args);        
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }
}

TEST(MessageSerializerTest, UdfDefTest) {
  {
    MemGuard mem_guard;
    UdfChainRegister::UdfDef udf;
    udf.inputs.emplace_back(std::make_pair(0, 1));
    udf.inputs.emplace_back(std::make_pair(2, 3));
    udf.udf_name = "test_udf";
    WrapperData<UdfChainRegister::UdfDef>* data = 
      new WrapperData<UdfChainRegister::UdfDef>(udf);

    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<UdfChainRegister::UdfDef>* r = 
      dynamic_cast<WrapperData<UdfChainRegister::UdfDef>*>(result);    

    EXPECT_EQ(string("test_udf"), r->Internal().udf_name);
    EXPECT_EQ((size_t)2, r->Internal().inputs.size());
    EXPECT_EQ(0, r->Internal().inputs[0].first);
    EXPECT_EQ(1, r->Internal().inputs[0].second);
    EXPECT_EQ(2, r->Internal().inputs[1].first);
    EXPECT_EQ(3, r->Internal().inputs[1].second);
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }
}

TEST(MessageSerializerTest, UdfChainRegisterTest) {
  {
    MemGuard mem_guard;
    UdfChainRegister::UdfDef udf;
    udf.inputs.emplace_back(std::make_pair(0, 1));
    udf.inputs.emplace_back(std::make_pair(2, 3));
    udf.udf_name = "test_udf";

    UdfChainRegister udf_chain;
    udf_chain.udfs.emplace_back(udf);
    udf_chain.hash = 0x0000000011111111;
    udf_chain.outputs.emplace_back(std::make_pair(0, 1));
    udf_chain.outputs.emplace_back(std::make_pair(2, 3));

    WrapperData<UdfChainRegister>* data = 
      new WrapperData<UdfChainRegister>(udf_chain);

    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    ps::Data* result = nullptr;    
    size_t len;
    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);
    EXPECT_TRUE(DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard).IsOk());
    WrapperData<UdfChainRegister>* r = dynamic_cast<WrapperData<UdfChainRegister>*>(result);    

    EXPECT_EQ(0x0000000011111111UL, r->Internal().hash);

    EXPECT_EQ(string("test_udf"), r->Internal().udfs[0].udf_name);
    EXPECT_EQ((size_t)2, r->Internal().udfs[0].inputs.size());
    EXPECT_EQ(0, r->Internal().udfs[0].inputs[0].first);
    EXPECT_EQ(1, r->Internal().udfs[0].inputs[0].second);
    EXPECT_EQ(2, r->Internal().udfs[0].inputs[1].first);
    EXPECT_EQ(3, r->Internal().udfs[0].inputs[1].second);

    EXPECT_EQ(0, r->Internal().outputs[0].first);
    EXPECT_EQ(1, r->Internal().outputs[0].second);
    EXPECT_EQ(2, r->Internal().outputs[1].first);
    EXPECT_EQ(3, r->Internal().outputs[1].second);
    delete data;
    delete result;
    delete[] deserialize_buf.base;
  }
}

TEST(MessageSerializerTest, VecStringTest) {
  {
    MemGuard mem_guard;
    using StringVec = std::vector<std::string>;
    WrapperData<StringVec>* data = new WrapperData<StringVec>();
    data->Internal().push_back("this");
    data->Internal().push_back("is");
    data->Internal().push_back("a");
    data->Internal().push_back("test");
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);

    ps::Data* result = nullptr;    
    size_t len;
    Status s = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    EXPECT_TRUE(s.IsOk());
    WrapperData<StringVec>* r = dynamic_cast<WrapperData<StringVec>*>(result);    
    EXPECT_TRUE(r != nullptr);
    const StringVec& ret = r->Internal();
    EXPECT_EQ(4, ret.size());
    EXPECT_EQ("this", ret[0]);
    EXPECT_EQ("is", ret[1]);
    EXPECT_EQ("a", ret[2]);
    EXPECT_EQ("test", ret[3]);
    delete data;
    delete result;
  }
}

TEST(MessageSerializerTest, VecDoubleTest) {
  {
    MemGuard mem_guard;
    using DoubleVec = std::vector<double>;
    WrapperData<DoubleVec>* data = new WrapperData<DoubleVec>();
    data->Internal().push_back(1.67);
    data->Internal().push_back(-2.31);
    data->Internal().push_back(0);
    data->Internal().push_back(-0.0000001);
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);

    ps::Data* result = nullptr;    
    size_t len;
    Status s = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    EXPECT_TRUE(s.IsOk());
    WrapperData<DoubleVec>* r = dynamic_cast<WrapperData<DoubleVec>*>(result);    
    EXPECT_TRUE(r != nullptr);
    const DoubleVec& ret = r->Internal();
    EXPECT_EQ(4, ret.size());
    EXPECT_EQ(1.67, ret[0]);
    EXPECT_EQ(-2.31, ret[1]);
    EXPECT_EQ(0, ret[2]);
    EXPECT_FLOAT_EQ(-0.0000001, ret[3]);
    delete data;
    delete result;
  }
}

TEST(MessageSerializerTest, VecBoolTest) {
  {
    MemGuard mem_guard;
    using BoolVec = std::vector<bool>;
    WrapperData<BoolVec>* data = new WrapperData<BoolVec>();
    data->Internal().push_back(true);
    data->Internal().push_back(false);
    data->Internal().push_back(false);
    data->Internal().push_back(true);
    data->Internal().push_back(true);
    data->Internal().push_back(true);    
    size_t id;
    std::vector<Fragment> bufs;
    EXPECT_TRUE(SerializeAny<Data>(data, &id, &bufs, mem_guard).IsOk());

    Fragment deserialize_buf;
    FragmentConcat(bufs, &deserialize_buf);

    ps::Data* result = nullptr;    
    size_t len;
    Status s = DeserializeAny<Data>(id, &deserialize_buf, 0, &result, &len, mem_guard);
    EXPECT_TRUE(s.IsOk());
    WrapperData<BoolVec>* r = dynamic_cast<WrapperData<BoolVec>*>(result);    
    EXPECT_TRUE(r != nullptr);
    const BoolVec& ret = r->Internal();
    EXPECT_EQ(6, ret.size());
    EXPECT_EQ(true, ret[0]);
    EXPECT_EQ(false, ret[1]);
    EXPECT_EQ(false, ret[2]);
    EXPECT_EQ(true, ret[3]);
    EXPECT_EQ(true, ret[4]);
    EXPECT_EQ(true, ret[5]);    
    delete data;
    delete result;
  }
}
