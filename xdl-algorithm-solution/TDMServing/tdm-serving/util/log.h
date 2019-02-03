/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TDM_SERVING_UTIL_LOG_H_
#define TDM_SERVING_UTIL_LOG_H_

#include "glog/logging.h"

#define LOG_CONFIG(app_name, log_path, log_level)      \
  do {                                                      \
    google::InitGoogleLogging(app_name);                    \
    google::SetLogDestination(google::INFO, log_path);      \
    google::SetLogDestination(google::WARNING, log_path);   \
    google::SetLogDestination(google::ERROR, log_path);     \
    google::SetLogDestination(google::FATAL, log_path);     \
    FLAGS_minloglevel = log_level;                          \
  } while (0)

#define LOG_DEBUG DLOG(INFO)
#define LOG_INFO LOG(INFO)
#define LOG_WARN LOG(WARNING)
#define LOG_ERROR LOG(ERROR)

#endif  // TDM_SERVING_UTIL_LOG_H_
