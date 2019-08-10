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

#ifndef PS_COMMON_FILESYSTEM_H_
#define PS_COMMON_FILESYSTEM_H_

#include "ps-plus/common/status.h"
#include "ps-plus/common/plugin.h"
#include "tbb/concurrent_vector.h"

#include <vector>
#include <string>

namespace ps {

class FileSystem {
 private:
  static constexpr size_t BUFFER_SIZE = 1048576;
 public:
  class ReadStream {
   public:
    ReadStream() : close_(false), buffer_ptr_(-1), buffer_size_(-1) {}
    virtual ~ReadStream() {Close();}
    Status Read(void* buf, size_t size);
    Status ReadBuffer();
    virtual int64_t ReadSimple(void* buf, size_t size) = 0;
    void Close();
    Status Eof(bool* eof);

    template <typename T>
    Status ReadRaw(T* data);
    template <typename T>
    Status ReadVec(std::vector<T>* data);
    template <typename T>    
    Status ReadTBBVec(tbb::concurrent_vector<T>* data);
    Status ReadStr(std::string* data);
    Status ReadShortStr(std::string* data);
   protected:
    virtual void CloseInternal() = 0;
   private:
    bool close_;
    char buffer_[BUFFER_SIZE];
    int buffer_ptr_;
    int buffer_size_;
  };

  class WriteStream {
   public:
    WriteStream() : close_(false) {}
    virtual ~WriteStream() {Close();}
    Status Write(const void* buf, size_t size);
    Status WriteBuffer();
    virtual int64_t WriteSimple(const void* buf, size_t size) = 0;
    virtual void Flush() = 0;
    void Close();

    template <typename T>
    Status WriteRaw(T data);
    template <typename T>
    Status WriteVec(const std::vector<T>& data);
    template <typename T>
    Status WriteTBBVec(const tbb::concurrent_vector<T>& data);
    Status WriteStr(const std::string& data);
    Status WriteShortStr(const std::string& data);
   protected:
    virtual void CloseInternal() = 0;
   private:
    bool close_;
  };

  virtual ~FileSystem() {}

  virtual Status OpenReadStream(const std::string& name, ReadStream** result) = 0;
  virtual Status OpenWriteStream(const std::string& name, WriteStream** result, bool append = false) = 0;
  virtual Status Mkdir(const std::string& name) = 0;
  virtual Status ListDirectory(const std::string& name, std::vector<std::string>* results) = 0;
  virtual Status Remove(const std::string& name) = 0;
  virtual Status Rename(const std::string& src_name, const std::string& dst_name) = 0;

  Status OpenReadStream(const std::string& name, std::unique_ptr<ReadStream>* result) {
    ReadStream* out;
    PS_CHECK_STATUS(OpenReadStream(name, &out));
    result->reset(out);
    return Status::Ok();
  }

  Status OpenWriteStream(const std::string& name, std::unique_ptr<WriteStream>* result, bool append = false) {
    WriteStream* out;
    PS_CHECK_STATUS(OpenWriteStream(name, &out, append));
    result->reset(out);
    return Status::Ok();
  }

  static Status GetFileSystem(const std::string& name, FileSystem** fs);
  static Status OpenReadStreamAny(const std::string& name, ReadStream** result);
  static Status OpenWriteStreamAny(const std::string& name, WriteStream** result, bool append = false);
  static Status OpenReadStreamAny(const std::string& name, std::unique_ptr<ReadStream>* result);
  static Status OpenWriteStreamAny(const std::string& name, std::unique_ptr<WriteStream>* result, bool append = false);
  static Status MkdirAny(const std::string& dir);
  static Status ListDirectoryAny(const std::string& dir, std::vector<std::string>* files);
  static Status RemoveAny(const std::string& name);
  static Status RenameAny(const std::string& src_name, const std::string& dst_name);
};

template <typename T>
Status FileSystem::ReadStream::ReadRaw(T* data) {
  return Read(data, sizeof(T));
}

template <typename T>
Status FileSystem::ReadStream::ReadVec(std::vector<T>* data) {
  size_t size;
  PS_CHECK_STATUS(ReadRaw(&size));
  data->resize(size);
  return Read((&(*data)[0]), sizeof(T) * size);
}

template <typename T>
Status FileSystem::ReadStream::ReadTBBVec(tbb::concurrent_vector<T>* data) {
  size_t size;
  PS_CHECK_STATUS(ReadRaw(&size));
  data->resize(size);
  for (size_t i = 0; i < data->size(); i++) {
    PS_CHECK_STATUS(Read((&(*data)[i]), sizeof(T)));
  }
  return Status::Ok();
}

template <typename T>
Status FileSystem::WriteStream::WriteRaw(T data) {
  return Write(&data, sizeof(T));
}

template <typename T>
Status FileSystem::WriteStream::WriteVec(const std::vector<T>& data) {
  size_t size = data.size();
  PS_CHECK_STATUS(WriteRaw(size));
  return Write((&data[0]), sizeof(T) * size);
}

template <typename T>
Status FileSystem::WriteStream::WriteTBBVec(const tbb::concurrent_vector<T>& data) {
  size_t size = data.size();
  PS_CHECK_STATUS(WriteRaw(size));
  for (size_t i = 0; i < data.size(); i++) {
    PS_CHECK_STATUS(Write((&data[i]), sizeof(T)));
  }
  return Status::Ok();
}
}

#endif

