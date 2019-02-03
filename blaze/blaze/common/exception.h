/*
 * \file exception.h 
 * \brief The blaze exception.
 */
#pragma once

#include <sstream>

#include "blaze/common/common_defines.h"

namespace blaze {

inline void MakeStringInternal(std::stringstream& /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
std::string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

extern void SetStackTraceFetcher(std::function<std::string(void)> fetcher); 

class Exception : public std::exception {
 public:
  Exception(const char* file,
            const int line,
            const char* condition,
            const std::string& msg,
            const void* caller = nullptr);
  
  void AppendMessage(const std::string& msg);
  std::string msg() const;
  inline const std::vector<std::string>& msg_stack() const {
    return msg_stack_;
  }

  const char* what() const noexcept override;
  const void* caller() const noexcept;

 protected:
  std::vector<std::string> msg_stack_;
  std::string full_msg_;
  std::string stack_trace_;
  const void* caller_;
};

#ifndef BLAZE_CONDITION_THROW
#define BLAZE_CONDITION_THROW(condition, ...)            \
    do { \
      if (!(condition)) { \
        throw blaze::Exception(__FILE__, __LINE__, #condition, MakeString(__VA_ARGS__)); \
      } \
    } while (false)
#endif

#ifndef BLAZE_THROW
#define BLAZE_THROW(...) \
    throw blaze::Exception(__FILE__, __LINE__, "", MakeString(__VA_ARGS__))
#endif

#ifndef CHECK
#define CHECK(condition, ...) if (!(condition)) BLAZE_THROW("[", __FILE__, ":", __LINE__, "] ", ##__VA_ARGS__)
#endif

#ifndef CHECK_TRUE
#define CHECK_TRUE(condition, ...) if(!(condition)) BLAZE_THROW((condition), ##__VA_ARGS__);
#endif

#ifndef CHECK_EQ
#define CHECK_EQ(a, b, ...) if ((a) != (b)) BLAZE_THROW("not equal ", ##__VA_ARGS__);
#endif

#ifndef CHECK_NE
#define CHECK_NE(a, b, ...) if ((a) == (b)) BLAZE_THROW("equal ", ##__VA_ARGS__);
#endif

}  // namespace blaze

