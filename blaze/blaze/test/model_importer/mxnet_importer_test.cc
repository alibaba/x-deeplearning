/*
 * \file mxnet_importer_test.cc
 * \brief The mxnet importer test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/model_importer/mxnet_importer.h"

namespace blaze {

TEST(TestMXNetImporter, LoadModel1) {
  MXNetImporter mxnet_importer;
  mxnet_importer.LoadModel("./utest_data/model_importer/mxnet/dnn.json",
                           "./utest_data/model_importer/mxnet/dnn.params");
  mxnet_importer.SaveToTextFile("dnn.blaze.mx.conf");
}

TEST(TestMXNetImporter, LoadModel2) {
  MXNetImporter mxnet_importer;
  mxnet_importer.LoadModel("./utest_data/model_importer/mxnet/din.json",
                           "./utest_data/model_importer/mxnet/din.params");
  mxnet_importer.SaveToTextFile("din.blazei.mx.conf");
}

}  // namespace blaze


