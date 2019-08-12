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

#include "xdl/core/utils/logging.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <stdlib.h>
#include <time.h>

#include "xdl/core/utils/time_utils.h"

namespace xdl {

LogMessage::LogMessage(const char* fname, int line, int severity)
  : fname_(fname), line_(line), severity_(severity) {}

void LogMessage::GenerateLogMessage() {
  uint64_t now_micros = TimeUtils::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
           localtime(&now_seconds));
  fprintf(stderr, "%s.%06d: %c %s:%d] %s\n", time_buffer, micros_remainder,
          "DIWEF"[severity_], fname_, line_, str().c_str());
}

namespace {
int64_t GetLogLevelFromEnv() {
  const char* log_level = getenv("XDL_CPP_LOG_LEVEL");
  if (log_level == nullptr) {
    return INFO;
  }

  return atoi(log_level);
}
}

LogMessage::~LogMessage() {
  static int64_t min_log_level = GetLogLevelFromEnv();
  if (likely(severity_ >= min_log_level)) {
    GenerateLogMessage();
  }
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}

LogMessageFatal::~LogMessageFatal() {
  GenerateLogMessage();
  abort();
}

} // namespace xdl
