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

#ifndef TDM_SERVING_UTIL_CONCURRENCY_CHECK_ERROR_H_
#define TDM_SERVING_UTIL_CONCURRENCY_CHECK_ERROR_H_

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace tdm_serving {
namespace util {

inline void HandleErrnoError(const char* function_name, int error) {
  const char* msg = strerror(error);
  fprintf(stderr, "%s: Fatal error, %s", function_name, msg);
  abort();
}

inline void CheckErrnoError(const char* function_name, int error) {
  if (error) {
    HandleErrnoError(function_name, error);
  }
}

inline void HandlePosixError(const char* function_name) {
  HandleErrnoError(function_name, errno);
}

inline void CheckPosixError(const char* function_name, int result) {
  if (result < 0) {
    HandlePosixError(function_name);
  }
}

inline bool HandlePosixTimedError(const char* function_name) {
  int error = errno;
  if (error == ETIMEDOUT) {
    return false;
  }
  HandleErrnoError(function_name, error);
  return true;
}

inline bool CheckPosixTimedError(const char* function_name, int result) {
  if (result < 0) {
    return HandlePosixTimedError(function_name);
  }
  return true;
}

inline bool HandlePthreadTimedError(const char* function_name, int error) {
  if (error == ETIMEDOUT) {
    return false;
  }
  HandleErrnoError(function_name, error);
  return false;
}

inline bool CheckPthreadTimedError(const char* function_name, int error) {
  if (error) {
    return HandlePthreadTimedError(function_name, error);
  }
  return true;
}

inline bool HandlePthreadTryLockError(const char* function_name, int error) {
  if (error == EBUSY || error == EAGAIN) {
    return false;
  }
  HandleErrnoError(function_name, error);
  return false;
}

inline bool CheckPthreadTryLockError(const char* function_name, int error) {
  if (error) {
    return HandlePthreadTryLockError(function_name, error);
  }
  return true;
}

#define CHECK_ERRNO_ERROR(expr) \
    CheckErrnoError(__PRETTY_FUNCTION__, (expr))

#define CHECK_POSIX_ERROR(expr) \
    CheckPosixError(__PRETTY_FUNCTION__, (expr))

#define CHECK_POSIX_TIMED_ERROR(expr) \
    CheckPosixTimedError(__PRETTY_FUNCTION__, (expr))

#define CHECK_PTHREAD_ERROR(expr) \
    CHECK_ERRNO_ERROR((expr))

#define CHECK_PTHREAD_TIMED_ERROR(expr) \
    CheckPthreadTimedError(__PRETTY_FUNCTION__, (expr))

#define CHECK_PTHREAD_TRYLOCK_ERROR(expr) \
    CheckPthreadTryLockError(__PRETTY_FUNCTION__, (expr))

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_CONCURRENCY_CHECK_ERROR_H_
