/*
 * \file model_importer_test.cc
 * \brief The model importer test unit
 */
#include "gtest/gtest.h"

#include <thread>

#include "blaze/model_importer/model_importer.h"

namespace blaze {

TEST(TestModelImporter, set_data_type) {
  ModelImporter model_importer;
  model_importer.set_data_type(kFloat);
  EXPECT_EQ(kFloat, model_importer.data_type());
}

TEST(TestModelImporter, set_weight_type) {
  ModelImporter model_importer;
  model_importer.set_weight_type(kFloat16);
  EXPECT_EQ(kFloat16, model_importer.weight_type());
}

TEST(TestModelImporter, set_op_weight_type) {
  ModelImporter model_importer;
  model_importer.set_op_weight_type("Gemm", kFloat16);
  EXPECT_EQ(model_importer.op_weight_type("Gemm"), kFloat16);
}

}  // namespace blaze
