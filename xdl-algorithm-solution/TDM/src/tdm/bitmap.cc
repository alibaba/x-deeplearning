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

// Copyright 2018 Alibaba Inc. All Rights Reserved.

#include "tdm/bitmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>

namespace tdm {

Bitmap::Bitmap(): data_(NULL), capacity_(0) {
}

Bitmap::Bitmap(const std::string& filename): data_(NULL), capacity_(0) {
  Load(filename);
}

bool Bitmap::Load(const std::string& filename) {
  FILE* fp = fopen(filename.c_str(), "r");
  if (fp == NULL) {
    return false;
  }

  if (fseek(fp, 0, SEEK_END) == -1) {
    fclose(fp);
    return false;
  }

  int64_t file_len = ftell(fp);
  if (file_len == -1) {
    fclose(fp);
    return false;
  }

  data_ = malloc(file_len);
  if (data_ == NULL) {
    fclose(fp);
    return false;
  }

  fseek(fp, 0, SEEK_SET);
  if (fread(data_, 1, file_len, fp) < file_len) {
    free(data_);
    fclose(fp);
    return false;
  }

  capacity_ = file_len;
  fclose(fp);
  return true;
}

bool Bitmap::is_filtered(size_t index) const {
  if (data_ == NULL || index >= capacity_) {
    return false;
  }
  uint64_t* ptr = reinterpret_cast<uint64_t*>(data_);
  uint64_t mask = 1;
  mask <<= index % 64;
  return ptr[index / 64] & mask;
}

bool Bitmap::set_filter(size_t index, bool filter) {
  if (index >= capacity_) {
    size_t cap = 1;
    while (cap <= index) {
      cap <<= 1;
    }

    void* new_data = NULL;
    if ((new_data = malloc(cap)) == NULL) {
      return false;
    }

    if (data_ != NULL && capacity_ > 0) {
      memcpy(new_data, data_, capacity_);
    }

    std::swap(data_, new_data);
    capacity_ = cap;
    free(new_data);
  }

  uint64_t* ptr = reinterpret_cast<uint64_t*>(data_);
  uint64_t mask = 1;
  mask <<= index % 64;

  if (filter) {
    ptr[index / 64] |= mask;
  } else {
    ptr[index / 64] &= ~mask;
  }

  return true;
}

bool Bitmap::save(const char* filename) const {
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    return false;
  }

  if (data_ != NULL && capacity_ > 0) {
    if (fwrite(data_, 1, capacity_, fp) < capacity_) {
      fclose(fp);
      return false;
    }
  }

  fclose(fp);
  return true;
}

}  // namespace tdm
