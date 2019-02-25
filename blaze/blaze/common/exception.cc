/*
 * \file exception.cc 
 * \brief The blaze exception.
 */
#include "blaze/common/exception.h"

#include <algorithm>

namespace blaze {

static std::function<std::string(void)> FetchStackTrace = []() { return ""; };

void SetStackTraceFetcher(std::function<std::string(void)> fetcher) {
  FetchStackTrace = fetcher;
}

std::string StripBasename(const std::string &full_path) {
  const char kSeparator = '/';
  size_t pos = full_path.rfind(kSeparator);
  if (pos != std::string::npos) {
    return full_path.substr(pos + 1, std::string::npos);
  } else {
    return full_path;
  }
}

Exception::Exception(const char* file,
                     const int line,
                     const char* condition,
                     const std::string& msg,
                     const void* caller) :
    msg_stack_{MakeString("[failed at ",
                          StripBasename(std::string(file)),
                          ":",
                          line,
                          "]",
                          condition,
                          ". ",
                          msg,
                          " ")},
    stack_trace_(FetchStackTrace()) {
  caller_ = caller;
  full_msg_ = this->msg();
}

void Exception::AppendMessage(const std::string& msg) {
  msg_stack_.push_back(msg);
  full_msg_ = this->msg();
}

std::string Exception::msg() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), std::string("")) + stack_trace_;
}

const char* Exception::what() const noexcept {
  return full_msg_.c_str();
}

const void* Exception::caller() const noexcept {
  return caller_;
}

}  // namespace blaze

