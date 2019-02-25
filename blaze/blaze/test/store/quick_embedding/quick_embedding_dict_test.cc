/*!
 * \file quick_embedding_dict_test.cc
 * \brief The quick embedding dict test unit
 */

#include "blaze/store/quick_embedding/quick_embedding_dict.h"

#include <fstream>

#include "thirdparty/gtest/gtest.h"

namespace blaze {
namespace store {

Trie::Key table1 = "101";
Trie::Key table2 = "102";
Trie::Key table3 = "104_7";
const int dim1 = 4;
const int dim2 = 8;
const int dim3 = 6;
HashTable::Key key1 = 1032342412;
HashTable::Key key2 = 849234344343;
HashTable::Key key3 = 3434234243;
HashTable::Key key4 = 5230242044;
float case1[dim1] = {1.0f, 1.0f, 3.0f, 1.0f};
float case2[dim2] = {0.15f, 0.3f, 2.0f, 1.0f, 1.0f, 2.0f, 0.23f, 3.0f};
float case3[dim3] = {2.5f, 1.0f, 0.15f, 0.3f, 2.0f, 1.0f};
float case4[dim3] = {1.0f, 0.15f, 0.3f, 2.0f, 1.0f, 1.0f};

void MockQuickEmbedding(const std::string& url) {
  VersionVerifier verifier(DictValueType::fp32);
  BulkLoadTrie trie;
  BulkLoadHashTable hashtable[kMaxGidSize];
  WeightBlob<float> weight_blob[kMaxGidSize];

  trie.PreInsert(table1, 0);
  trie.PreInsert(table2, 1);
  trie.PreInsert(table3, 2);
  trie.BulkLoad();

  weight_blob[0].AllocateMemory(sizeof(float) * dim1);
  weight_blob[1].AllocateMemory(sizeof(float) * dim2);
  weight_blob[2].AllocateMemory(sizeof(float) * dim3 * 2);

  // insert case
  float* weights = nullptr;
  uint64_t offset = weight_blob[0].InsertWeights(dim1, &weights);
  EXPECT_TRUE(weights != nullptr);
  memcpy(weights, case1, sizeof(float) * dim1);
  hashtable[0].PreInsert(key1, offset);

  offset = weight_blob[1].InsertWeights(dim2, &weights);
  EXPECT_TRUE(weights != nullptr);
  memcpy(weights, case2, sizeof(float) * dim2);
  hashtable[1].PreInsert(key2, offset);

  offset = weight_blob[2].InsertWeights(dim3, &weights);
  EXPECT_TRUE(weights != nullptr);
  memcpy(weights, case3, sizeof(float) * dim3);
  hashtable[2].PreInsert(key3, offset);

  offset = weight_blob[2].InsertWeights(dim3, &weights);
  EXPECT_TRUE(weights != nullptr);
  memcpy(weights, case4, sizeof(float) * dim3);
  hashtable[2].PreInsert(key4, offset);

  // write bin file
  std::ofstream os(url, std::ios::binary);
  EXPECT_TRUE(os.is_open());
  EXPECT_TRUE(verifier.Dump(&os));
  EXPECT_TRUE(trie.Dump(&os));
  for (uint16_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(os.write((char*)&i, sizeof(i)));
    EXPECT_TRUE(hashtable[i].BulkLoad());
    EXPECT_TRUE(hashtable[i].Dump(&os));
    EXPECT_TRUE(weight_blob[i].Dump(&os));
  }
  os.close();
}

TEST(TestQuickEmbeddingDict, SelfCheck) {
  QuickEmbeddingDict dict;
  const std::string fake_url = "no_exist";
  EXPECT_FALSE(dict.SelfCheck(fake_url));
  const std::string url = "test.ut.quickembedding.bin";
  MockQuickEmbedding(url);
  EXPECT_TRUE(dict.SelfCheck(url));
}


TEST(TestQuickEmbeddingDict, Load) {
  QuickEmbeddingDict dict;
  const std::string fake_url = "no_exist";
  EXPECT_EQ(dict.Load(fake_url), kFail);
  const std::string url = "test.ut.quickembedding.bin";
  MockQuickEmbedding(url);
  EXPECT_EQ(dict.Load(url), kOK);
}

TEST(TestQuickEmbeddingDict, Get) {
  const std::string url = "test.ut.quickembedding.bin";
  QuickEmbeddingDict dict;
  EXPECT_EQ(dict.Load(url), kOK);

  std::vector<SparsePullerInput> sparse_puller_inputs;
  std::vector<SparsePullerOutput> sparse_puller_outputs;

  float out_block1[24];
  float out_block2[48];

  // fg1 input
  SparsePullerInput fg1_input;
  fg1_input.name = table1;
  int64_t fg1_keys[2] = {1032342412, 12312422};
  fg1_input.key = reinterpret_cast<void*>(fg1_keys);
  int32_t fg1_key_nums[2] = {1, 1};
  fg1_input.key_num = reinterpret_cast<void*>(fg1_key_nums);
  fg1_input.key_num_size = 2;
  float fg1_values[2] = {2.0f, 1.2f};
  fg1_input.value = reinterpret_cast<void*>(fg1_values);
  fg1_input.key_type = blaze::kInt64;
  fg1_input.num_type = blaze::kInt32;
  fg1_input.value_type = blaze::kFloat;
  fg1_input.dim = dim1;
  SparsePullerInput::Param in_param;
  in_param.udf_type = UDFType::kSum;
  fg1_input.in_item.push_back(in_param);
  sparse_puller_inputs.push_back(fg1_input);
  // fg1 output
  SparsePullerOutput fg1_output;
  SparsePullerOutput::OutItem out_param;
  out_param.out = out_block1;
  out_param.stride = dim1 + dim2;
  fg1_output.out_item.push_back(out_param);
  sparse_puller_outputs.push_back(fg1_output);

  // fg2 input
  SparsePullerInput fg2_input;
  fg2_input.name = table2;
  int64_t fg2_keys[3] = {2324222, 849234344343, 849234344343};
  fg2_input.key = reinterpret_cast<void*>(fg2_keys);
  int32_t fg2_key_nums[2] = {1, 2};
  fg2_input.key_num = reinterpret_cast<void*>(fg2_key_nums);
  fg2_input.key_num_size = 2;
  float fg2_values[3] = {2.0f, 1.2f, 1.0f};
  fg2_input.value = reinterpret_cast<void*>(fg2_values);
  fg2_input.key_type = blaze::kInt64;
  fg2_input.num_type = blaze::kInt32;
  fg2_input.value_type = blaze::kFloat;
  fg2_input.dim = dim2;
  in_param.udf_type = UDFType::kSum;
  fg2_input.in_item.push_back(in_param);
  sparse_puller_inputs.push_back(fg2_input);
  // fg2 output
  SparsePullerOutput fg2_output;
  out_param.out = out_block1 + dim1;
  out_param.stride = dim1 + dim2;
  fg2_output.out_item.push_back(out_param);
  sparse_puller_outputs.push_back(fg2_output);

  // fg3 input
  SparsePullerInput fg3_input;
  fg3_input.name = table3;
  int64_t fg3_keys[6] = {3434234243, 5230242044, 2424123, 3434234243, 0, 5230242044};
  fg3_input.key = reinterpret_cast<void*>(fg3_keys);
  int32_t fg3_key_nums[2] = {4, 2};
  fg3_input.key_num = reinterpret_cast<void*>(fg3_key_nums);
  fg3_input.key_num_size = 2;
  float fg3_values[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  fg3_input.value = reinterpret_cast<void*>(fg3_values);
  fg3_input.key_type = blaze::kInt64;
  fg3_input.num_type = blaze::kInt32;
  fg3_input.value_type = blaze::kFloat;
  fg3_input.dim = dim3;
  in_param.udf_type = UDFType::kAssign;
  in_param.trunc_direction = TruncDirection::kOrder;
  in_param.trunc_num = 4;
  fg3_input.in_item.push_back(in_param);
  sparse_puller_inputs.push_back(fg3_input);
  // fg3 output
  SparsePullerOutput fg3_output;
  out_param.out = out_block2;
  out_param.stride = 24;
  fg3_output.out_item.push_back(out_param);
  sparse_puller_outputs.push_back(fg3_output);

  EXPECT_EQ(dict.Get(sparse_puller_inputs, sparse_puller_outputs), kOK);

  // compare block1 result
  for (int i = 0; i < 4; ++i) {
    float expect_value = case1[i] * fg1_values[0];
    float actual_value = out_block1[i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }
  for (int i = 0; i < 8; ++i) {
    float expect_value = 0.0f;
    float actual_value = out_block1[dim1 + i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }
  for (int i = 0; i < 4; ++i) {
    float expect_value = case2[i] * (fg2_values[1] + fg2_values[2]);
    float actual_value = out_block1[dim1 + dim2 + dim1 + i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }

  // compare block2 result
  for (int i = 0; i < 6; ++i) {
    float expect_value = case3[i] * fg3_values[0];
    float actual_value = out_block2[i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }
  for (int i = 0; i < 6; ++i) {
    float expect_value = case4[i] * fg3_values[1];
    float actual_value = out_block2[dim3 + i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }
  for (int i = 0; i < 6; ++i) {
    float expect_value = 0.0f;
    float actual_value = out_block2[dim3 * 2 + i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }
  for (int i = 0; i < 6; ++i) {
    float expect_value = case3[i] * fg3_values[3];
    float actual_value = out_block2[dim3 * 3 + i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }
  for (int i = 0; i < 6; ++i) {
    float expect_value = 0.0f;
    float actual_value = out_block2[dim3 * 4 + i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }
  for (int i = 0; i < 6; ++i) {
    float expect_value = case4[i] * fg3_values[5];
    float actual_value = out_block2[dim3 * 5 + i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }
  for (int i = 0; i < 12; ++i) {
    float expect_value = 0.0f;
    float actual_value = out_block2[dim3 * 6 + i];
    EXPECT_FLOAT_EQ(expect_value, actual_value);
  }

  /*
  for (int i = 0; i < 24; ++i) {
    LOG_INFO("%f", out_block1[i]);
  }
  for (int i = 0; i < 48; ++i) {
    LOG_INFO("%f", out_block2[i]);
  }*/
}

}  // namespace store
}  // namespace blaze