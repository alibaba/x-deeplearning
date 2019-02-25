/*
 * \file onnx_importer_test.cc
 * \brief The onnx importer test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/model_importer/onnx_importer.h"

namespace blaze {

TEST(TestOnnxImporter, TestDNN) {
  ONNXImporter onnx_importer;
  onnx_importer.LoadModel("./utest_data/onnx/dnn.onnx");

  auto ret = onnx_importer.SaveToTextFile("dnn.blaze.conf");
  EXPECT_TRUE(ret);

  ret = onnx_importer.SaveToBinaryFile("dnn.blaze.dat");
  EXPECT_TRUE(ret);
}

TEST(TestOnnxImporter, TestDin) {
  ONNXImporter onnx_importer;
  onnx_importer.LoadModel("./utest_data/onnx/din.onnx");

  auto ret = onnx_importer.SaveToTextFile("din.blaze.conf");
  EXPECT_TRUE(ret);

  ret = onnx_importer.SaveToBinaryFile("din.blaze.dat");
  EXPECT_TRUE(ret);
}

TEST(TestOnnxImporter, TestAttention) {
  ONNXImporter onnx_importer;
  onnx_importer.LoadModel("./utest_data/onnx/attention.onnx");

  auto ret = onnx_importer.SaveToTextFile("attention.blaze.conf");
  EXPECT_TRUE(ret);

  ret = onnx_importer.SaveToBinaryFile("attention.blaze.dat");
  EXPECT_TRUE(ret);
}

}  // namespace blaze
