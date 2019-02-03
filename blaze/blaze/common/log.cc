/*
 * \file log.cc 
 * \brief The blaze logging module. 
 */
#include "blaze/common/log.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <unistd.h>
#include <sys/syscall.h>

#include <time.h>
#include <stdarg.h>
#include <string>

namespace blaze {

Logger::Logger(LogLevel level) : level_(level), file_(nullptr) { }

Logger::Logger(std::string filename, LogLevel level) : level_(level), file_(nullptr) {
  ResetLogFile(filename);
}

Logger::~Logger() {
  CloseLogFile();
}

int Logger::ResetLogFile(std::string filename) {
  CloseLogFile();
  if (filename.size() > 0) {
    file_ = fopen(filename.c_str(), "w");
    if (file_ == nullptr) {
      Write(LogLevel::kError, "Cannot create log file %s\n", filename.c_str());
      return -1;
    }
  }
  return 0;
}

void Logger::Write(LogLevel level, const char* format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(level, format, &val);
  va_end(val);
}

void Logger::Debug(const char* format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(LogLevel::kInfo, format, &val);
  va_end(val);
}

void Logger::Info(const char* format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(LogLevel::kInfo, format, &val);
  va_end(val);
}

void Logger::Error(const char* format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(LogLevel::kError, format, &val);
  va_end(val);
}

void Logger::Fatal(const char* format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(LogLevel::kFatal, format, &val);
  va_end(val);
}

void Logger::WriteImpl(LogLevel level, const char* format, va_list* val) {
  if (level >= level_) {
    std::string level_str = GetLevelStr(level);
    std::string time_str = GetSystemTime();
    va_list val_copy;
    va_copy(val_copy, *val);

    fprintf(file_ ? file_ : stderr, "[%s] [%s] [%d] ", level_str.c_str(), time_str.c_str(), syscall(SYS_gettid));
    vfprintf(file_ ? file_ : stderr, format, val_copy);
    fprintf(file_ ? file_ : stderr, "\n");
    fflush(file_ ? file_ : stderr);

    va_end(val_copy);
    if (level == LogLevel::kFatal) {
      abort();
    }
  }
}

void Logger::CloseLogFile() {
  if (file_ != nullptr) {
    fclose(file_);
    file_ = nullptr;
  }
}

std::string Logger::GetSystemTime() {
  time_t t = time(0);
  char str[64];
  strftime(str, sizeof(str), "%Y-%m-%d %H:%M:%S", localtime(&t));
  return str;
}

std::string Logger::GetLevelStr(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kInfo:
      return "INFO ";
    case LogLevel::kError:
      return "ERROR";
    case LogLevel::kFatal:
      return "FATAL";
    default:
      return "UNKOWN";
  }
}

Logger Log::logger_;

int Log::ResetLogFile(std::string filename) {
  return logger_.ResetLogFile(filename);
}

void Log::ResetLogLevel(LogLevel level) {
  logger_.ResetLogLevel(level);
}

void Log::Write(LogLevel level, const char* format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(level, format, &val);
  va_end(val);
}

void Log::Debug(const char* format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(LogLevel::kDebug, format, &val);
  va_end(val);
}

void Log::Info(const char* format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(LogLevel::kInfo, format, &val);
  va_end(val);
}

void Log::Error(const char* format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(LogLevel::kError, format, &val);
  va_end(val);
}

void Log::Fatal(const char* format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(LogLevel::kFatal, format, &val);
  va_end(val);
}

}  // namespace blaze
