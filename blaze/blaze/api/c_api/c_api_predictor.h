/*
 * \file c_api_predictor.h
 */
#ifndef H_CAPI_PREDICTOR_H_
#define H_CAPI_PREDICTOR_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* PredictorManagerHandle;
typedef void* PredictorHandle;

// Init Scheduler
int Blaze_InitScheduler(int enable_batching,
                        int max_batch_size,
                        int batch_timeout_micros,
                        int num_threads_for_cpu,
                        int num_threads_for_cuda,
                        int num_threads_for_pipe);

// Load model of PredictorManager
int Blaze_PredictorManagerCreate(PredictorManagerHandle* handle);
int Blaze_PredictorManagerDelete(PredictorManagerHandle handle);
int Blaze_PredictorManagerSetDataType(PredictorManagerHandle handle,
                                      int data_type);
int Blaze_PredictorManagerSetRunMode(PredictorManagerHandle handle,
                                      const char* run_mode);

int Blaze_LoadSparseModelWeight(PredictorManagerHandle handle,
                                const char* sparse_model_weight_file);
int Blaze_PredcitorManagerLoadModel(PredictorManagerHandle handle,
                                    const char* blaze_model_file,
                                    int optimization_pass);
int Blaze_PredictorManagerLoadDeepNetModel(PredictorManagerHandle handle,
                                           const char* deepnet_conf_file,
                                           const char* deepnet_param_file,
                                           int optimization_pass);

// Predicator creation
int Blaze_PredictorCreate(PredictorManagerHandle predictor_mgr,
                          int device_type,
                          int device_id,
                          PredictorHandle* handle);
int Blaze_PredictorDelete(PredictorHandle handle);

// Predictor operations
// about input
int Blaze_PredictorReshapeInput(PredictorHandle handle,
                                const char* name,
                                const int* data,
                                int num);
int Blaze_PredictorFeed(PredictorHandle handle,
                        const char* name,
                        void* data,
                        int len);
int Blaze_PredictorForward(PredictorHandle handle);

// about output
int Blaze_PredictorOutputShape(PredictorHandle handle,
                               const char* name,
                               size_t* ndim,
                               size_t** shape);
int Blaze_PredictorOutputDataType(PredictorHandle handle,
                                  const char* name,
                                  int* data_type);
int Blaze_PredictorOutput(PredictorHandle handle,
                          const char* name,
                          void* data,
                          size_t len);

// about internal param
int Blaze_PredictorParamName(PredictorHandle handle,
                             size_t* ndim,
                             const char*** name);
int Blaze_PredictorParamShape(PredictorHandle handle,
                              const char* name,
                              size_t* ndim,
                              size_t** shape);
int Blaze_PredictorParamDataType(PredictorHandle handle,
                                 const char* name,
                                 int* data_type);
int Blaze_PredictorParam(PredictorHandle handle,
                         const char* name,
                         void* data,
                         size_t len);

// about List input and output names
int Blaze_PredictorListInputNames(PredictorHandle handle,
                                  size_t* ndim,
                                  const char*** names);
int Blaze_PredictorListOutputNames(PredictorHandle handle,
                                   size_t* ndim,
                                   const char*** names);

// about observers
int Blaze_PredictorRegisterObservers(PredictorHandle handle,
                                     size_t ndim,
                                     const char** names);
int Blaze_PredictorDumpObservers(PredictorHandle handle,
                                 size_t* ndim,
                                 const char*** key,
                                 const char*** value);

#ifdef __cplusplus
}
#endif

#endif  // H_CAPI_PREDICTOR_H_

