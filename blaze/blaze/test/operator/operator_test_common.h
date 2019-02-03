/*
 * \file operator_test_common.h
 * \brief The operator test common
 */
#include "gtest/gtest.h"

#include <math.h>
#include <thread>

#include "blaze/common/proto_helper.h"
#include "blaze/operator/operator.h"
#include "blaze/graph/workspace.h"
#include "blaze/graph/net.h"
#include "blaze/store/sparse_puller.h"

namespace blaze {

// Now only support cpu mode.
template <typename T, bool strict_check=true>
void TestOperatorOutput(const char* net_conf_path, const std::vector<std::string>& output_str,
                        const char* sparse_db_uri = "", const char* sparse_db_type = "") {
  Workspace workspace;
  // set sparse model weight
  if (strlen(sparse_db_type) != 0) {
    std::shared_ptr<SparsePuller> sparse_puller;
    sparse_puller.reset(SparsePullerCreationRegisterer::Get()->CreateSparsePuller(sparse_db_type));
    sparse_puller->Load(sparse_db_uri);
    workspace.SetSparsePuller(sparse_puller);
  }
  //workspace.SetSparseDbConfig(sparse_db_uri, sparse_db_type);
  std::shared_ptr<Net> net = workspace.CreateNet(net_conf_path);
  EXPECT_TRUE(net->Run());

  const BlazeMap<std::string, Blob*>& blob_map = net->external_output_blob();
  // EXPECT_EQ(output_str.size(), blob_map.size());
  const NetDef& net_def = net->net_def();
  ArgumentHelper argument_helper(net_def);

  DeviceOption host_device_option;
  host_device_option.set_device_type(0);
  host_device_option.set_device_id(0);

  for (const auto& str : output_str) {
    EXPECT_TRUE(blob_map.find(str) != blob_map.end());
    Blob* blob = blob_map.find(str)->second;
    const std::vector<TIndex>& shape = blob->shape();

    Blob cpu_blob(host_device_option, blob->shape(), static_cast<DataType>(TypeFlag<T>::kFlag));
    Copy(&cpu_blob, blob, 0);
    if (blob->device_type() == kCUDA) {
#ifdef USE_CUDA
      cudaStreamSynchronize(0);
#endif
    }

    // Check output shape
    std::string shape_key = str + "_shape";
    std::vector<TIndex> expected_shape = argument_helper.GetRepeatedArgument<TIndex>(shape_key);
    EXPECT_EQ(expected_shape.size(), shape.size());
    for (size_t k = 0; k < shape.size(); ++k) {
      EXPECT_EQ(expected_shape[k], shape[k]);
    }
    // Check output value
    std::vector<T> expected_data = argument_helper.GetRepeatedArgument<T>(str);
    for (size_t k = 0; k < blob->size(); ++k) {
      bool check_equal = false;
      if (strict_check) {
        check_equal = (fabs(cpu_blob.as<T>()[k] - expected_data[k]) <= 1e-6);
      } else {
        // check ralative diff
        float x1 = cpu_blob.as<T>()[k];
        float x2 = expected_data[k];
        if (fabs(x1 - x2) / (1e-8 + std::max(fabs(x1), fabs(x2))) <= 0.05) {
          check_equal = true;
        }
      }
      if (!check_equal) {
        LOG_INFO("k=%d data=%f expected_data=%f", k, (float)cpu_blob.as<T>()[k], (float)expected_data[k]);
      } 
      EXPECT_TRUE(check_equal);
    }
  }
  LOG_INFO("Testing Operator Output Success -> %s", net_conf_path);
}

}
