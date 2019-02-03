/*
 * \file typeid.cc
 * \desc The typeid utility.
 */
#include "blaze/common/typeid.h"

#include <cxxabi.h>

namespace blaze {

std::string Demangle(const char* name) {
  int status;
  auto demangled = ::abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (demangled) {
    std::string ret;
    ret.assign(demangled);
    free(demangled);
    return ret;
  }
  return name;
}

}  // namespace blaze
