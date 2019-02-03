/*
 * \file xdl_importer_test.cc
 * \brief The xdl importer test unit
 */
#include "gtest/gtest.h"

#include "blaze/model_importer/xdl_importer.h"

namespace blaze {

TEST(TestXdlImporter, TestLoadModel) {
  XdlImporter xdl_importer;
  xdl_importer.LoadModel(
      "./utest_data/model_importer/xdl/graph.txt",
      "./utest_data/model_importer/xdl/dense.txt");

  xdl_importer.SaveToTextFile("xdl.blaze.txt");
  xdl_importer.SaveToBinaryFile("xdl.blaze.bin");
}

}  // namespace blaze
