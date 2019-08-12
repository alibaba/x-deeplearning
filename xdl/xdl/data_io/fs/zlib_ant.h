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

#ifndef XDL_IO_FS_ZLIB_ANT_H_
#define XDL_IO_FS_ZLIB_ANT_H_

#include "xdl/core/lib/common_defines.h"
#include "xdl/data_io/fs/file_system.h"
#include "xdl/data_io/fs/zlib_compression_options.h"
#include <memory>

namespace xdl {
namespace io {

class ZlibAnt : public IOAnt {
 public:
  ZlibAnt(IOAnt* input_stream, const ZlibCompressionOptions& zlib_options = ZlibCompressionOptions::DEFAULT());
  virtual ~ZlibAnt();
  virtual ssize_t Read(char* data, size_t len);
  virtual ssize_t Write(const char* data, size_t len);
  virtual off_t Seek(off_t offset);
  DISALLOW_COPY_AND_ASSIGN(ZlibAnt);
 private:
  void InitZlibBuffer();
  ssize_t ReadFromStream();
  void Inflate();
  size_t ReadBytesFromCache(size_t bytes_to_read, char* result);
  size_t NumUnreadBytes() const;
  
  std::unique_ptr<IOAnt> input_stream_;
  size_t input_buffer_capacity_;        // Size of z_stream_input_
  size_t output_buffer_capacity_;       // Size of z_stream_output_
  char* next_unread_byte_;              // Next unread byte in z_stream_output_
  std::unique_ptr<Bytef[]> z_stream_input_;
  std::unique_ptr<Bytef[]> z_stream_output_;
  ZlibCompressionOptions const zlib_options_;
  std::unique_ptr<z_stream> z_stream_;
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_FS_FILE_SYSTEM_H_