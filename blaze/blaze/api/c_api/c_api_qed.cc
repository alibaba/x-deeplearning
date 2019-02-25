/*
 * \file c_api_qed.cc
 */
#include "blaze/api/c_api/c_api_qed.h"

#include "blaze/store/quick_embedding/embedding_builder.h"

using blaze::store::EmbeddingBuilder;
using blaze::float16;

int Blaze_QuickEmbeddingBuildFp32(const char* path,
                                  const char* meta,
                                  const char* output_file,
                                  int thread_num) {
  EmbeddingBuilder<float> builder;
  try {
    builder.Build(path, meta, output_file, thread_num);
  } catch (std::exception& e) {
    LOG_ERROR("failed: %s", e.what());
    return -1;
  }
  return 0;
}

int Blaze_QuickEmbeddingBuildFp16(const char* path,
                                  const char* meta,
                                  const char* output_file,
                                  int thread_num) {
  EmbeddingBuilder<float16> builder;
  try {
    builder.Build(path, meta, output_file, thread_num);
  } catch (std::exception& e) {
    LOG_ERROR("failed: %s", e.what());
    return -1;
  }
  return 0;
}
