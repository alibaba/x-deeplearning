/*
 * \file predictor_test.cc
 * \brief The predictor test module
 */
#include "blaze/api/cpp_api/predictor.h"

#include "gtest/gtest.h"

namespace blaze {

TEST(TestPredictorManager, LoadDeepNet) {
  PredictorManager predictor_manager;
  auto ret = predictor_manager.LoadDeepNetModel(
      "./utest_data/model_importer/ulf/gauss_cnxh_din_v2/net-parameter-conf",
      "./utest_data/model_importer/ulf/gauss_cnxh_din_v2/dnn-model-dat",
      true);
  EXPECT_TRUE(ret);
}

TEST(TestPredictor, TestSparseFeatureName2FeedName) {
  std::string feed_name = FeedNameUtility::SparseFeatureName2FeedName("item_id", kSparseFeatureId);
  EXPECT_STREQ("item_id.ids", feed_name.c_str());
}

TEST(TestPredictor, TestIndicatorLevel2FeedName) {
  std::string feed_name = FeedNameUtility::IndicatorLevel2FeedName(0);
  EXPECT_STREQ("indicator.0", feed_name.c_str());
}

}  // namespace blaze
