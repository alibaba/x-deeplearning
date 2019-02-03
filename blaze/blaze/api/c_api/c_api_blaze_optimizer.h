/*
 * \file c_api_blaze_optimizer.h
 */
#ifndef H_CAPI_BLAZE_OPTIMIZER_H_
#define H_CAPI_BLAZE_OPTIMIZER_H_

#ifdef __cplusplus
extern "C" {
#endif

// Convert mxnet model to blaze model
int Blaze_OptimizeBlaze(const char* raw_model_file,
                        const char* dst_model_file,
                        int binary);

#ifdef __cplusplus
}
#endif

#endif  // H_CAPI_BLAZE_OPTIMIZER_H_
