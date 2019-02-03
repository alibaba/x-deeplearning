/*
 * \file c_api_qed.h
 */
#ifndef H_CAPI_QED_H_
#define H_CAPI_QED_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int Blaze_QuickEmbeddingBuildFp32(const char* path,
                                  const char* meta,
                                  const char* output_file,
                                  int thread_num);

int Blaze_QuickEmbeddingBuildFp16(const char* path,
                                  const char* meta,
                                  const char* output_file,
                                  int thread_num);

#ifdef __cplusplus
}
#endif

#endif  // H_CAPI_QED_H_
