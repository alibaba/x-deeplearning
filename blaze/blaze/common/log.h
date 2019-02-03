/*
 * \file log.h 
 * \brief The blaze logging module. 
 */
#pragma once

#include <string>

namespace blaze {

enum LogLevel {
  kDebug = 0,
  kInfo,
  kError,
  kFatal,
};

class Logger {
  friend class Log;

 public:
  explicit Logger(LogLevel level = LogLevel::kInfo);
  explicit Logger(std::string filename, LogLevel level = LogLevel::kInfo);
  ~Logger();

  int ResetLogFile(std::string filename);
  void ResetLogLevel(LogLevel level) { level_ = level; }

  void Write(LogLevel level, const char* format, ...);
  void Debug(const char* format, ...);
  void Info(const char* format, ...);
  void Error(const char* format, ...);
  void Fatal(const char* format, ...);

 private:
  void WriteImpl(LogLevel level, const char* format, va_list* val);
  void CloseLogFile();
  std::string GetSystemTime();
  std::string GetLevelStr(LogLevel level);

  FILE* file_;
  LogLevel level_;
};

class Log {
 public:
  static int ResetLogFile(std::string filename);
  static void ResetLogLevel(LogLevel level);
  static std::string GetLevelStr(LogLevel level);

  static void Write(LogLevel level, const char* format, ...);
  static void Debug(const char* format, ...);
  static void Info(const char* format, ...);
  static void Error(const char* format, ...);
  static void Fatal(const char* format, ...);

 private:
  static Logger logger_;
};

#ifndef LOG_SET_FILE
#define LOG_SET_FILE(filename) blaze::Log::ResetLogFile(filename)
#endif

#ifndef LOG_SET_LEVEL
#define LOG_SET_LEVEL(level)   blaze::Log::ResetLogLevel(level)
#endif

#ifndef LOG_DEBUG
#define LOG_DEBUG(format, ...) blaze::Log::Debug("[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#ifndef LOG_INFO
#define LOG_INFO(format, ...)  blaze::Log::Info("[%s:%d] "  format, __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(format, ...) blaze::Log::Error("[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#ifndef LOG_FATAL
#define LOG_FATAL(format, ...) blaze::Log::Fatal("[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__)
#endif

}  // namespace blaze
