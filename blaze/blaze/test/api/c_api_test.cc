/*
 * \file c_api_test.cc
 * \brief The c api test module
 */
#include "blaze/api/c_api/c_api_predictor.h"
#include "blaze/api/c_api/c_api_blaze_converter.h"
#include "blaze/common/log.h"
#include "blaze/api/cpp_api/predictor.h"
#include "blaze/proto/blaze.pb.h"

#include <math.h>
#include "gtest/gtest.h"

namespace blaze {

TEST(TestONNX2Blaze, Onnx2Blaze) {
  auto ret = Blaze_ConvertBlaze("./utest_data/onnx/dnn.onnx",
                                "",
                                kOnnx,
                                kFloat,
                                kFloat,
                                0,
                                nullptr,
                                nullptr,
                                "./dnn.onnx2blaze",
                                0);
  EXPECT_EQ(0, ret);

  ret = Blaze_ConvertBlaze("./utest_data/onnx/dnn.onnx",
                           "",
                           kOnnx,
                           kFloat,
                           kFloat,
                           0,
                           nullptr,
                           nullptr,
                           "./dnn.onnx2blaze",
                           1);
  EXPECT_EQ(0, ret);
}

TEST(TestPreditorManager, Create) {
  PredictorManagerHandle handle;
  auto ret = Blaze_PredictorManagerCreate(&handle);
  EXPECT_EQ(0, ret);
  EXPECT_TRUE(handle != nullptr);

  ret = Blaze_PredictorManagerDelete(handle);
  EXPECT_EQ(0, ret);
}

TEST(TestPreditorManager, Load) {
  PredictorManagerHandle handle;
  auto ret = Blaze_PredictorManagerCreate(&handle);
  ret = Blaze_PredictorManagerSetDataType(handle, 1);
  EXPECT_EQ(0, ret);
  
  ret = Blaze_PredictorManagerSetRunMode(handle, "simple");
  EXPECT_EQ(0, ret);

  ret = Blaze_PredcitorManagerLoadModel(handle, "./dnn.onnx2blaze", 1);
  EXPECT_EQ(0, ret);

  // create handle
  PredictorHandle predict_handle = nullptr;
  ret = Blaze_PredictorCreate(handle, 0, 0, &predict_handle);
  EXPECT_EQ(0, ret);
  EXPECT_TRUE(predict_handle != nullptr);

  // list input names
  const char** names;
  size_t dim;
  ret = Blaze_PredictorListInputNames(predict_handle, &dim, &names);
  EXPECT_EQ(0, ret);
  for (auto i = 0; i < dim; ++i) {
    LOG_INFO("name[%d]=%s", i, names[i]);
    EXPECT_TRUE(strcmp(names[i], "comm") == 0 ||
                strcmp(names[i], "ncomm") == 0);
  }

  /// Reshape input
  int comm_shape[] = { 200, 540 };
  int ncomm_shape[] = { 200, 360 };
  ret = Blaze_PredictorReshapeInput(predict_handle, "comm", comm_shape, 2);
  EXPECT_EQ(0, ret);
  ret = Blaze_PredictorReshapeInput(predict_handle, "ncomm", ncomm_shape, 2);
  EXPECT_EQ(0, ret);

  // Feed Input
  float comm_data[200*540] = { -0.00001 };
  float ncomm_data[200 * 360] = { 0.0001 };
  ret = Blaze_PredictorFeed(predict_handle, "comm", comm_data, sizeof(float) * 200 * 540);
  EXPECT_EQ(0, ret);
  ret = Blaze_PredictorFeed(predict_handle, "ncomm", ncomm_data, sizeof(float) * 200 * 360);
  EXPECT_EQ(0, ret);

  // Register Observers
  const char* observer_name[] = { "cost", "profile" };
  ret = Blaze_PredictorRegisterObservers(predict_handle, 2, observer_name);
  EXPECT_EQ(0, ret);

  // Forward
  ret = Blaze_PredictorForward(predict_handle);
  EXPECT_EQ(0, ret);

  // List output names
  const char** onames;
  size_t odims;
  ret = Blaze_PredictorListOutputNames(predict_handle, &odims, &onames);
  EXPECT_EQ(0, ret);
  EXPECT_EQ(1, odims);
  EXPECT_STREQ(onames[0], "out");

  // Output shape and type
  size_t* oshape;
  size_t odim;
  int odata_type;
  ret = Blaze_PredictorOutputShape(predict_handle, "out", &odim, &oshape);
  EXPECT_EQ(0, ret);
  ret = Blaze_PredictorOutputDataType(predict_handle, "out", &odata_type);
  EXPECT_EQ(0, ret);

  EXPECT_EQ(2, odim);
  EXPECT_EQ(200, oshape[0]);
  EXPECT_EQ(2, oshape[1]);
  EXPECT_EQ(1/*float32*/, odata_type);

  // Fetch output
  float odata[200 * 2];
  ret = Blaze_PredictorOutput(predict_handle, "out", odata, 400 * sizeof(float));
  EXPECT_EQ(0, ret);
  for (auto i = 0; i < 200; ++i) {
    LOG_INFO("%f %f", odata[i * 2], odata[i * 2 + 1]);
  }

  // Get Internal param names
  size_t internal_ndim;
  const char** internal_names;
  ret = Blaze_PredictorParamName(predict_handle, &internal_ndim, &internal_names);
  EXPECT_EQ(0, ret);
  for (auto i = 0; i < internal_ndim; ++i) {
    LOG_INFO("internal_name[%d]=%s", i, internal_names[i]);
  }

  // Get each internal names's info
  bool check_first = true;
  for (auto i = 0; i < internal_ndim; ++i) {
    auto iname = internal_names[i];
    // Get internal param shape and type.
    size_t indim;
    size_t* inshape;
    ret = Blaze_PredictorParamShape(predict_handle, iname, &indim, &inshape);
    EXPECT_EQ(0, ret);
    int idata_type;
    ret = Blaze_PredictorParamDataType(predict_handle, iname, &idata_type);
    EXPECT_EQ(1, idata_type);
    EXPECT_EQ(0, ret);
    // get param data.
    size_t len = 1;
    for (auto k = 0; k < indim; ++k) len *= inshape[k];
    float *data = new float[len];
    ret = Blaze_PredictorParam(predict_handle, iname, data, sizeof(float) * len);
    EXPECT_EQ(0, ret);
    // display data
    for (auto k = 0; k < len; ++k) {
      if (isnan(data[k]) && check_first) {
        check_first = false;
        LOG_INFO("name=%s data[%d]=%f", iname, k, data[k]);
      }
    }
    delete [] data;
  }

  // Dump Observers
  size_t ob_ndim;
  const char** ob_key;
  const char** ob_value;
  ret = Blaze_PredictorDumpObservers(predict_handle, &ob_ndim, &ob_key, &ob_value);
  EXPECT_EQ(0, ret);
  for (auto i = 0; i < ob_ndim; ++i) {
    LOG_INFO("key=%s value=%s", ob_key[i], ob_value[i]);
  }

  // Destruction
  ret = Blaze_PredictorDelete(predict_handle);
  EXPECT_EQ(0, ret);
  ret = Blaze_PredictorManagerDelete(handle);
  EXPECT_EQ(0, ret);
}

}  // namespace blaze
