/*
 * \file c_api_blaze_converter.h
 */
#ifndef H_CAPI_BLAZE_CONVERTER_H_
#define H_CAPI_BLAZE_CONVERTER_H_

#ifdef __cplusplus
extern "C" {
#endif

// Convert mxnet model to blaze model
int Blaze_ConvertBlaze(const char* model_conf_file,
                       const char* model_data_file,
                       int model_type,
                       int weight_type,
                       int io_type,
                       int x_type_num,
                       const char** x_type_key,
                       const int* x_type_value,
                       const char* blaze_model_file,
                       int binary);

#ifdef __cplusplus
}
#endif

#endif  // H_CAPI_BLAZE_CONVERSION_H_
