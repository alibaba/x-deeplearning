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

#ifndef PS_COMMON_LOGGING_H_
#define PS_COMMON_LOGGING_H_

#include <limits>
#include <sstream>

namespace ps {
const int DEBUG = 0;         
const int INFO = 1;          
const int WARNING = 2;       
const int ERROR = 3;         
const int FATAL = 4;         
const int NUM_SEVERITIES = 5;

class LogMessage : public std::basic_ostringstream<char> {
 public:
  LogMessage(const char* fname, int line, int severity);
  ~LogMessage();

 protected:
  void GenerateLogMessage();

 private:
  const char* fname_;
  int line_;
  int severity_;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line);
  ~LogMessageFatal();
};

#define _PS_LOG_INFO \
  ::ps::LogMessage(__FILE__, __LINE__, ::ps::INFO)
#define _PS_LOG_DEBUG \
  ::ps::LogMessage(__FILE__, __LINE__, ::ps::DEBUG)
#define _PS_LOG_WARNING \
  ::ps::LogMessage(__FILE__, __LINE__, ::ps::WARNING)
#define _PS_LOG_ERROR \
  ::ps::LogMessage(__FILE__, __LINE__, ::ps::ERROR)
#define _PS_LOG_FATAL \
  ::ps::LogMessageFatal(__FILE__, __LINE__)

#define LOG(severity) _PS_LOG_##severity

#define DLOG(severity) PS_LOG(severity)

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define PS_CHECK(condition)                    \
  if (unlikely(!(condition)))                   \
    LOG(FATAL) << "Check failed: " #condition " "

#define PS_CHECK_EQ(lhs, rhs)                          \
  if (unlikely(((lhs) != (rhs))))                       \
    LOG(FATAL) << "Check failed: " #lhs " == " #rhs


#define DCHECK(condition) PS_CHECK(condition)

}  // ps

#endif  // PS_COMMON_LOGGING_H_
