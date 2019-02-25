/*!
 * \file udf_processor_test.cc
 * \brief The udf processor test unit
 */
#include "blaze/store/quick_embedding/udf_processor.h"

#include "thirdparty/gtest/gtest.h"

namespace blaze {
namespace store {

TEST(Test, UdfProcessor) {
  UdfProcessor<float, float> processor;
  int dim = 12;
  float weights[12];
  for (int i = 0; i < dim; ++i) {
    weights[i] = (float) i;
  }
  size_t index = 1;
  size_t size = 1;
  SparsePullerInput::Param param;
  param.udf_type = UDFType::kSum;
  float out[12];
  EXPECT_TRUE(processor.InitProcess(dim, param, out));
  EXPECT_TRUE(processor.ElementProcess(1.0f,
                                       weights,
                                       dim,
                                       index,
                                       size,
                                       param,
                                       out));
  EXPECT_TRUE(processor.ReduceProcess(dim,
                                      size,
                                      param,
                                      out));
}

TEST(Test, KSumProcessor) {
  UdfProcessor<float, float> *processor =
      UdfProcessorFactory<float, float>::Create(UDFType::kSum);
  int dim = 12;
  float out[12];
  size_t size = 10;
  SparsePullerInput::Param param;
  param.udf_type = UDFType::kSum;
  float value = 0.5f;
  // init process
  EXPECT_TRUE(processor->InitProcess(dim, param, out));
  for (auto i = 0; i < dim; ++i) {
    EXPECT_FLOAT_EQ(0.0f, out[i]);
  }
  // element process do ksum
  for (auto i = 0; i < size; ++i) {
    float weights[12];
    for (auto j = 0; j < dim; ++j) {
      weights[j] = (float) j;
    }

    EXPECT_TRUE(processor->ElementProcess(value,
                                          weights,
                                          dim,
                                          i,
                                          size,
                                          param,
                                          out));
  }

  for (int j = 0; j < dim; ++j) {
    EXPECT_FLOAT_EQ(0.5 * j * 10, out[j]);
  }
  // reduce do nothing
  EXPECT_TRUE(processor->ReduceProcess(dim,
                                       size,
                                       param,
                                       out));
  for (int j = 0; j < dim; ++j) {
    EXPECT_FLOAT_EQ(value * j * 10, out[j]);
  }
}

TEST(Test, KAvgProcessor) {
  UdfProcessor<float, float> *processor =
      UdfProcessorFactory<float, float>::Create(UDFType::kAvg);
  int dim = 12;
  float out[12];
  size_t size = 10;
  SparsePullerInput::Param param;
  param.udf_type = UDFType::kAvg;
  float value = 0.5f;
  // init process
  EXPECT_TRUE(processor->InitProcess(dim, param, out));
  for (auto i = 0; i < dim; ++i) {
    EXPECT_FLOAT_EQ(0.0f, out[i]);
  }
  // element process do ksum
  for (auto i = 0; i < size; ++i) {
    float weights[12];
    for (auto j = 0; j < dim; ++j) {
      weights[j] = (float) j;
    }

    EXPECT_TRUE(processor->ElementProcess(value,
                                          weights,
                                          dim,
                                          i,
                                          size,
                                          param,
                                          out));
  }

  for (auto j = 0; j < dim; ++j) {
    EXPECT_FLOAT_EQ(0.5 * j * size, out[j]);
  }
  // reduce do avg
  EXPECT_TRUE(processor->ReduceProcess(dim,
                                       size,
                                       param,
                                       out));
  for (auto j = 0; j < dim; ++j) {
    EXPECT_FLOAT_EQ(0.5 * j, out[j]);
  }
}

TEST(TestKAssignProcessor, OrderTrunc) {
  UdfProcessor<float, float> *processor =
      UdfProcessorFactory<float, float>::Create(UDFType::kAssign);
  int dim = 12;
  float out[60];
  size_t size = 10;
  SparsePullerInput::Param param;
  param.udf_type = UDFType::kAssign;
  param.trunc_num = 5;
  param.trunc_direction = TruncDirection::kOrder;
  float value = 0.5f;
  // init process
  EXPECT_TRUE(processor->InitProcess(dim, param, out));
  for (auto i = 0; i < dim * param.trunc_num; ++i) {
    EXPECT_FLOAT_EQ(0.0f, out[i]);
  }
  // element process do assign
  for (auto i = 0; i < size; ++i) {
    float weights[12];
    for (auto j = 0; j < dim; ++j) {
      weights[j] = 0.1 * i + j;
    }

    EXPECT_TRUE(processor->ElementProcess(value,
                                          weights,
                                          dim,
                                          i,
                                          size,
                                          param,
                                          out));
  }

  int offset = 0;
  for (auto i = 0; i < param.trunc_num; ++i) {
    for (auto j = 0; j < dim; ++j) {
      float expect_value = value * (0.1 * i + j);
      EXPECT_FLOAT_EQ(expect_value, out[j + offset]);
    }
    offset += dim;
  }
  // reduce do nothing
  EXPECT_TRUE(processor->ReduceProcess(dim,
                                       size,
                                       param,
                                       out));
  offset = 0;
  for (auto i = 0; i < param.trunc_num; ++i) {
    for (auto j = 0; j < dim; ++j) {
      float expect_value = value * (0.1 * i + j);
      EXPECT_FLOAT_EQ(expect_value, out[j + offset]);
    }
    offset += dim;
  }
}

TEST(TestKAssignProcessor, ReverseTrunc) {
  UdfProcessor<float, float> *processor =
      UdfProcessorFactory<float, float>::Create(UDFType::kAssign);
  int dim = 12;
  float out[60];
  size_t size = 10;
  SparsePullerInput::Param param;
  param.udf_type = UDFType::kAssign;
  param.trunc_num = 4;
  param.trunc_direction = TruncDirection::kReverse;
  float value = 0.5f;
  // init process
  EXPECT_TRUE(processor->InitProcess(dim, param, out));
  for (auto i = 0; i < dim * param.trunc_num; ++i) {
    EXPECT_FLOAT_EQ(0.0f, out[i]);
  }
  // element process do assign
  for (auto i = 0; i < size; ++i) {
    float weights[12];
    for (auto j = 0; j < dim; ++j) {
      weights[j] = 0.1 * i + j;
    }

    EXPECT_TRUE(processor->ElementProcess(value,
                                          weights,
                                          dim,
                                          i,
                                          size,
                                          param,
                                          out));
  }

  int offset = 0;
  for (auto i = size - param.trunc_num; i < size; ++i) {
    for (auto j = 0; j < dim; ++j) {
      float expect_value = value * (0.1 * i + j);
      EXPECT_FLOAT_EQ(expect_value, out[j + offset]);
    }
    offset += dim;
  }
  // reduce do nothing
  EXPECT_TRUE(processor->ReduceProcess(dim,
                                       size,
                                       param,
                                       out));
  offset = 0;
  for (auto i = size - param.trunc_num; i < size; ++i) {
    for (auto j = 0; j < dim; ++j) {
      float expect_value = value * (0.1 * i + j);
      EXPECT_FLOAT_EQ(expect_value, out[j + offset]);
    }
    offset += dim;
  }
}

TEST(TestKAssignProcessor, NoTrunc) {
  UdfProcessor<float, float> *processor =
      UdfProcessorFactory<float, float>::Create(UDFType::kAssign);
  UdfProcessorFactory<float, float>::Create(UDFType::kAssign);
  int dim = 12;
  float out[240];
  size_t size = 10;
  SparsePullerInput::Param param;
  param.udf_type = UDFType::kAssign;
  param.trunc_num = 20;
  param.trunc_direction = TruncDirection::kReverse;
  float value = 0.5f;
  // init process
  EXPECT_TRUE(processor->InitProcess(dim, param, out));
  for (auto i = 0; i < dim * param.trunc_num; ++i) {
    EXPECT_FLOAT_EQ(0.0f, out[i]);
  }
  // element process do assign
  for (auto i = 0; i < size; ++i) {
    float weights[12];
    for (auto j = 0; j < dim; ++j) {
      weights[j] = 0.1 * i + j;
    }

    EXPECT_TRUE(processor->ElementProcess(value,
                                          weights,
                                          dim,
                                          i,
                                          size,
                                          param,
                                          out));
  }

  int offset = 0;
  float* current_out = out;
  current_out += dim * (param.trunc_num - size);
  for (auto i = 0; i < size; ++i) {
    for (auto j = 0; j < dim; ++j) {
      float expect_value = value * (0.1 * i + j);
      EXPECT_FLOAT_EQ(expect_value, current_out[j + offset]);
    }
    offset += dim;
  }
  // reduce do nothing
  EXPECT_TRUE(processor->ReduceProcess(dim,
                                       size,
                                       param,
                                       out));
  offset = 0;
  current_out = out;
  current_out += dim * (param.trunc_num - size);
  for (auto i = 0; i < size; ++i) {
    for (auto j = 0; j < dim; ++j) {
      float expect_value = value * (0.1 * i + j);
      EXPECT_FLOAT_EQ(expect_value, current_out[j + offset]);
    }
    offset += dim;
  }
}

}  // namespace store
}  // namespace blaze
