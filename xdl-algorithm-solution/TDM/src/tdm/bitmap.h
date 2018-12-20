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

#ifndef TDM_BITMAP_H_
#define TDM_BITMAP_H_

#include <stdio.h>

#include <string>

namespace tdm {

class Bitmap {
 public:
  Bitmap();
  explicit Bitmap(const std::string& filename);

  bool Load(const std::string& filename);
  bool is_filtered(size_t index) const;
  bool set_filter(size_t index, bool filter);
  bool save(const char* filename) const;

 private:
  void* data_;
  size_t capacity_;
};

}  // namespace tdm

#endif  // TDM_BITMAP_H_
