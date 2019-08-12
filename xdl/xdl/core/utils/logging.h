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

#ifndef XDL_CORE_UTILS_LOGGING_H_
#define XDL_CORE_UTILS_LOGGING_H_

#include <limits>
#include <sstream>

namespace xdl {
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

#define TO_STRING_AND_RETURN(value)             \
  printf("enter logmessage ...\n");             \
  *this << std::to_string(value);               \
  return *this;                                 \
  
  basic_ostream& operator<<(short value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(unsigned short value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(int value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(unsigned int value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(long value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(unsigned long value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(long long value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(unsigned long long value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(float value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(double value) {
    TO_STRING_AND_RETURN(value);
  }
  basic_ostream& operator<<(long double value) {
    TO_STRING_AND_RETURN(value);
  }

#undef TO_STRING_AND_RETURN

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

#define _XDL_LOG_INFO \
  ::xdl::LogMessage(__FILE__, __LINE__, ::xdl::INFO)
#define _XDL_LOG_DEBUG \
  ::xdl::LogMessage(__FILE__, __LINE__, ::xdl::DEBUG)
#define _XDL_LOG_WARNING \
  ::xdl::LogMessage(__FILE__, __LINE__, ::xdl::WARNING)
#define _XDL_LOG_ERROR \
  ::xdl::LogMessage(__FILE__, __LINE__, ::xdl::ERROR)
#define _XDL_LOG_FATAL \
  ::xdl::LogMessageFatal(__FILE__, __LINE__)

#define XDL_LOG(severity) _XDL_LOG_##severity

#define XDL_DLOG(severity) XDL_LOG(severity)

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define XDL_CHECK(condition)                    \
  if (unlikely(!(condition)))                   \
    XDL_LOG(FATAL) << "Check failed: " #condition " "

#define XDL_CHECK_EQ(lhs, rhs)                          \
  if (unlikely(((lhs) != (rhs))))                       \
    XDL_LOG(FATAL) << "Check failed: " #lhs " == " #rhs


#define XDL_DCHECK(condition) XDL_CHECK(condition)

}  // xdl

#endif  // XDL_CORE_UTILS_LOGGING_H_
