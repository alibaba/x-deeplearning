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

#include "xdl/data_io/fs/zlib_ant.h"
#include "xdl/core/utils/logging.h"
#include <iostream>

namespace xdl {
namespace io {

ZlibAnt::ZlibAnt(
    IOAnt* input_stream,
    const ZlibCompressionOptions& zlib_options)
    : input_stream_(input_stream),
      input_buffer_capacity_(zlib_options.input_buffer_size),
      output_buffer_capacity_(zlib_options.output_buffer_size),
      z_stream_input_(new Bytef[input_buffer_capacity_]),
      z_stream_output_(new Bytef[output_buffer_capacity_]),
      zlib_options_(zlib_options),
      z_stream_(new z_stream) {
  InitZlibBuffer();
}

ZlibAnt::~ZlibAnt() {
  if (z_stream_.get()) {
    inflateEnd(z_stream_.get());
  }
}

void ZlibAnt::InitZlibBuffer() {
  memset(z_stream_.get(), 0, sizeof(z_stream));

  z_stream_->zalloc = Z_NULL;
  z_stream_->zfree = Z_NULL;
  z_stream_->opaque = Z_NULL;
  z_stream_->next_in = Z_NULL;
  z_stream_->avail_in = 0;

  int status = inflateInit2(z_stream_.get(), zlib_options_.window_bits);
  if (status != Z_OK) {
    XDL_LOG(FATAL) << "inflateInit failed with status " << status;
    z_stream_.reset(NULL);
  } else {
    z_stream_->next_in = z_stream_input_.get();
    z_stream_->next_out = z_stream_output_.get();
    next_unread_byte_ = reinterpret_cast<char*>(z_stream_output_.get());
    z_stream_->avail_in = 0;
    z_stream_->avail_out = output_buffer_capacity_;
  }
}

ssize_t ZlibAnt::ReadFromStream() {
  int bytes_to_read = input_buffer_capacity_;
  char* read_location = reinterpret_cast<char*>(z_stream_input_.get());

  if (z_stream_->avail_in > 0) {
    uLong read_bytes = z_stream_->next_in - z_stream_input_.get();
    if (read_bytes > 0) {
      memmove(z_stream_input_.get(), z_stream_->next_in, z_stream_->avail_in);
    }

    bytes_to_read -= z_stream_->avail_in;
    read_location += z_stream_->avail_in;
  }
  ssize_t read_size = input_stream_->Read(read_location, bytes_to_read);
  if (read_size <= 0) {
    return read_size;
  }

  z_stream_->next_in = z_stream_input_.get();

  z_stream_->avail_in += read_size;
  return read_size;
}

size_t ZlibAnt::ReadBytesFromCache(size_t bytes_to_read, char* result) {
  size_t unread_bytes =
      reinterpret_cast<char*>(z_stream_->next_out) - next_unread_byte_;
  size_t can_read_bytes = std::min(bytes_to_read, unread_bytes);
  if (can_read_bytes > 0) {
    if (result != nullptr) {
      memcpy(result, next_unread_byte_, can_read_bytes);
    }
    next_unread_byte_ += can_read_bytes;
  }
  return can_read_bytes;
}

size_t ZlibAnt::NumUnreadBytes() const {
  size_t read_bytes =
      next_unread_byte_ - reinterpret_cast<char*>(z_stream_output_.get());
  return output_buffer_capacity_ - z_stream_->avail_out - read_bytes;
}

ssize_t ZlibAnt::Read(char* data, size_t bytes_to_read) {
  ssize_t origin_request_bytes = bytes_to_read;
  bytes_to_read -= ReadBytesFromCache(bytes_to_read, data);

  bool eof = false;
  while (bytes_to_read > 0 && !eof) {
    if (NumUnreadBytes() != 0) {
      return -1;
    }
    if (z_stream_->avail_in == 0) {
      ssize_t read_size = ReadFromStream();
      if (read_size < 0) {
        return read_size;
      } else if (read_size == 0) {
        eof = true;
      }
    }
    z_stream_->next_out = z_stream_output_.get();
    next_unread_byte_ = reinterpret_cast<char*>(z_stream_output_.get());
    z_stream_->avail_out = output_buffer_capacity_;

    Inflate();
    ssize_t ret = ReadBytesFromCache(bytes_to_read, data == nullptr ? nullptr
                                     : data + origin_request_bytes - bytes_to_read);
    bytes_to_read -= ret;
  }
  return origin_request_bytes - bytes_to_read;
}

void ZlibAnt::Inflate() {
  int error = inflate(z_stream_.get(), zlib_options_.flush_mode);
  if (error != Z_OK && error != Z_STREAM_END) {
    std::string error_string = "inflate() failed with error " + std::to_string(error);
    if (z_stream_->msg != NULL) {
      error_string += ": " + std::string(z_stream_->msg);
    }
    XDL_CHECK(false) << error_string;
  }
}

ssize_t ZlibAnt::Write(const char* data, size_t len) {
  return input_stream_->Write(data, len);
}

off_t ZlibAnt::Seek(off_t offset) {
  input_stream_->Seek(0);
  InitZlibBuffer();
  auto len = Read(nullptr, offset);
  XDL_CHECK(len == offset);
  return offset;
}

}
}
