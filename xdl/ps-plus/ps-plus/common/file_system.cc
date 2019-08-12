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

#include "ps-plus/common/file_system.h"

namespace ps {

constexpr size_t FileSystem::BUFFER_SIZE;

Status FileSystem::ReadStream::Read(void* buf, size_t size) {
  char* raw_buf = (char*)buf;
  size_t remain = size;
  while (remain > 0) {
    if (buffer_size_ == buffer_ptr_) {
      PS_CHECK_STATUS(ReadBuffer());
    }

    int64_t read = std::min((int64_t)remain, (int64_t)(buffer_size_ - buffer_ptr_));
    memcpy(raw_buf, buffer_ + buffer_ptr_, read);
    buffer_ptr_ += read;
    remain -= read;
    raw_buf += read;
  }
  return Status::Ok();
}

Status FileSystem::ReadStream::ReadBuffer() {
  if (buffer_size_ >= 0 && buffer_size_ < BUFFER_SIZE) {
    return Status::DataLoss("File exhausted");
  }
  char* raw_buf = buffer_;
  size_t remain = BUFFER_SIZE;
  size_t last = 0;
  int64_t read = 0;
  while (remain > 0) {
    int64_t read = ReadSimple(raw_buf, remain);
    if (read < 0 || (last == 0 && read == 0)) {
      break;
    }
    remain -= read;
    raw_buf += read;
    last = read;
  }
  if (read < 0) {
    return Status::DataLoss("File Read Error");
  }
  buffer_size_ = raw_buf - buffer_;
  buffer_ptr_ = 0;
  return Status::Ok();
}

Status FileSystem::ReadStream::Eof(bool* eof) {
  if (buffer_size_ == buffer_ptr_) {
    if (buffer_size_ >= 0 && buffer_size_ < BUFFER_SIZE) {
      *eof = true;
      return Status::Ok();
    } else {
      PS_CHECK_STATUS(ReadBuffer());
      *eof = buffer_size_ == 0;
      return Status::Ok();
    }
  }
  *eof = false;
  return Status::Ok();
}

void FileSystem::ReadStream::Close() {
  if (!close_) {
    CloseInternal();
    close_ = true;
  }
}

Status FileSystem::WriteStream::Write(const void* buf, size_t size) {
  const char* raw_buf = (const char*)buf;
  size_t remain = size;
  size_t last = 0;
  while (remain > 0) {
    int64_t write = WriteSimple(raw_buf, std::min(remain, BUFFER_SIZE));
    if (write < 0 || (last == 0 && write == 0)) {
      break;
    }
    remain -= write;
    raw_buf += write;
    last = write;
  }
  if (remain > 0) {
    return Status::Unknown("Some Error occured on write file");
  }
  return Status::Ok();
}

void FileSystem::WriteStream::Close() {
  if (!close_) {
    CloseInternal();
    close_ = true;
  }
}

Status FileSystem::GetFileSystem(const std::string& name, FileSystem** fs) {
  std::string protocol;
  size_t pos = name.find("://");
  if (pos == std::string::npos) {
    protocol = "file";
  } else {
    protocol = name.substr(0, pos);
  }
  *fs = GetPlugin<FileSystem>(protocol);
  if (*fs == nullptr) {
    return Status::NotFound("Filesystem [" + protocol + "] Not found");
  }
  return Status::Ok();
}

Status FileSystem::OpenReadStreamAny(const std::string& name, ReadStream** result) {
  FileSystem* fs;
  PS_CHECK_STATUS(GetFileSystem(name, &fs));
  return fs->OpenReadStream(name, result);
}

Status FileSystem::OpenWriteStreamAny(const std::string& name, WriteStream** result, bool append) {
  FileSystem* fs;
  PS_CHECK_STATUS(GetFileSystem(name, &fs));
  return fs->OpenWriteStream(name, result, append);
}

Status FileSystem::OpenReadStreamAny(const std::string& name, std::unique_ptr<ReadStream>* result) {
  FileSystem* fs;
  PS_CHECK_STATUS(GetFileSystem(name, &fs));
  return fs->OpenReadStream(name, result);
}

Status FileSystem::OpenWriteStreamAny(const std::string& name, std::unique_ptr<WriteStream>* result, bool append) {
  FileSystem* fs;
  PS_CHECK_STATUS(GetFileSystem(name, &fs));
  return fs->OpenWriteStream(name, result, append);
}

Status FileSystem::MkdirAny(const std::string& name) {
  FileSystem* fs;
  PS_CHECK_STATUS(GetFileSystem(name, &fs));
  return fs->Mkdir(name);
}

Status FileSystem::ListDirectoryAny(const std::string& name, std::vector<std::string>* results) {
  FileSystem* fs;
  PS_CHECK_STATUS(GetFileSystem(name, &fs));
  return fs->ListDirectory(name, results);
}

Status FileSystem::RemoveAny(const std::string& name) {
  FileSystem* fs;
  PS_CHECK_STATUS(GetFileSystem(name, &fs));
  return fs->Remove(name);
}

Status FileSystem::RenameAny(const std::string& src_name, const std::string& dst_name) {
  FileSystem* fs;
  PS_CHECK_STATUS(GetFileSystem(src_name, &fs));
  return fs->Rename(src_name, dst_name);
}

Status FileSystem::ReadStream::ReadStr(std::string* data) {
  size_t size;
  PS_CHECK_STATUS(ReadRaw(&size));
  data->resize(size);
  return Read((&(*data)[0]), sizeof(char) * size);
}

Status FileSystem::WriteStream::WriteStr(const std::string& data) {
  size_t size = data.size();
  PS_CHECK_STATUS(WriteRaw(size));
  return Write((&data[0]), sizeof(char) * size);
}

Status FileSystem::ReadStream::ReadShortStr(std::string* data) {
  int size;
  PS_CHECK_STATUS(ReadRaw(&size));
  data->resize(size);
  return Read((&(*data)[0]), sizeof(char) * size);
}

Status FileSystem::WriteStream::WriteShortStr(const std::string& data) {
  int size = data.size();
  PS_CHECK_STATUS(WriteRaw(size));
  return Write((&data[0]), sizeof(char) * size);
}

}

