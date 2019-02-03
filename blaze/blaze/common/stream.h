/*!
 * \file stream.h
 * \brief The I/O stream 
 */
#pragma once

#include <stdio.h>
#include <string>

namespace blaze {

/// Stream I/O for serialization
class Stream {
 public:
  /// Read data from stream
  virtual size_t Read(void *ptr, size_t size) = 0;
  /// Write data to stream
  virtual size_t Write(const void *ptr, size_t size) = 0;
  /// Empty
  virtual bool Empty() const = 0;
  /// Seek begin
  virtual void Seek(size_t offset = 0) = 0;
  /// Tell
  virtual size_t Tell() const = 0;
  /// Virtual destruction
  virtual ~Stream() { }

  /// Write data to stream
  template <typename T>
  inline void Write(const T& data) {
    Write(&data, sizeof(T));
  }
  /// Read data from stream
  template <typename T>
  inline size_t Read(T* out_data) {
    return Read(out_data, sizeof(T));
  }
};

/// FileStream I/O for serialization
class FileStream : public Stream {
 public:
  explicit FileStream(const std::string& file_name, bool read = false) {
    if (read) {
      fp_ = fopen(file_name.c_str(), "r");
    } else {
      fp_ = fopen(file_name.c_str(), "w");
    }
  }
  virtual ~FileStream() {
    if (fp_) fclose(fp_);
  }
  /// Read data from stream
  virtual size_t Read(void* ptr, size_t size) {
    return fread(ptr, 1, size, fp_);
  }
  /// Write data to stream
  virtual size_t Write(const void* ptr, size_t size) {
    return fwrite(ptr, 1, size, fp_);
  }
  /// Empty
  virtual bool Empty() const {
    return feof(fp_) != 0;
  }
  /// Seek begin
  virtual void Seek(size_t offset = 0) {
    fseek(fp_, offset, SEEK_SET);
  }
  /// Tell position
  virtual size_t Tell() const {
    return ftell(fp_);
  }
  bool Valid() {
    return fp_ != NULL;
  }

 protected:
  FILE* fp_;
};

}  // namespace blaze
