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


#ifndef XDL_IO_FS_ZLIB_COMPRESSION_OPTIONS_H_
#define XDL_IO_FS_ZLIB_COMPRESSION_OPTIONS_H_

#include <zlib.h>

namespace xdl {
namespace io {
class ZlibCompressionOptions {
 public:
  static ZlibCompressionOptions DEFAULT();
  static ZlibCompressionOptions RAW();
  static ZlibCompressionOptions GZIP();

  char flush_mode = Z_NO_FLUSH;
  int64_t input_buffer_size = 256 << 10;
  int64_t output_buffer_size = 256 << 10;
  char window_bits = MAX_WBITS;
  char compression_level = Z_DEFAULT_COMPRESSION;
  char compression_method = Z_DEFLATED;
  char mem_level = 9;
  char compression_strategy = Z_DEFAULT_STRATEGY;
};

inline ZlibCompressionOptions ZlibCompressionOptions::DEFAULT() {
  return ZlibCompressionOptions();
}

inline ZlibCompressionOptions ZlibCompressionOptions::RAW() {
  ZlibCompressionOptions options = ZlibCompressionOptions();
  options.window_bits = -options.window_bits;
  return options;
}

inline ZlibCompressionOptions ZlibCompressionOptions::GZIP() {
  ZlibCompressionOptions options = ZlibCompressionOptions();
  options.window_bits = options.window_bits + 16;
  return options;
}
}
}

#endif  //XDL_IO_FS_ZLIB_COMPRESSION_OPTIONS_H_