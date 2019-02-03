/*!
 * \file string_util.h
 * \brief The string utility
 */
#pragma once

#include <string>
#include <vector>

namespace blaze {

inline std::vector<std::string> Split(const std::string& raw_str, const char delim) {
  std::vector<std::string> splits;
  size_t i = 0, j = 0;
  for (; j < raw_str.length();) {
    if (raw_str.at(j) == delim) {
      splits.push_back(std::string(raw_str.c_str() + i, j - i));
      i = ++j;
    } else {
      ++j;
    }
  }
  splits.push_back(std::string(raw_str.c_str() + i, j - i));
  return splits;
}

inline std::vector<std::string> Split(const std::string& raw_str, const std::string& delim) {
  std::vector<std::string> splits;
  const char* s1 = raw_str.c_str();
  const char* s2 = delim.c_str();
  while (s1 != nullptr) {
    const char* pos = strstr(s1, s2);
    if (pos == nullptr) {
      splits.push_back(s1);
      break;
    }
    splits.push_back(std::string(s1, pos - s1));
    s1 = pos + delim.length();
  }
  return splits;
}

}  // namespace blaze

