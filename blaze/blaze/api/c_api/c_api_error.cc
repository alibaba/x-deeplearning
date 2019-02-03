/*
 * \file c_api_error.cc
 */
#include "blaze/api/c_api/c_api_error.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string>

#include "blaze/common/thread_local.h"

const int kMaxBufSize = 4 * 1024 * 1024;

struct StringEntry {
  char last_string[kMaxBufSize];
};

typedef blaze::ThreadLocalStore<StringEntry> ThreadLocalStringStore;

void Blaze_GetLastErrorString(const char** msg) {
  *msg = ThreadLocalStringStore::Get()->last_string;
}

void Blaze_SetLastErrorString(const char* format, ...) {
  va_list val;
  va_start(val, format);
  char* buf = ThreadLocalStringStore::Get()->last_string;
  int len = snprintf(buf, kMaxBufSize, format, val);
  buf[len] = '\0';
  va_end(val);
}
