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

#ifndef PS_PLUS_COMMON_HASHER_H_
#define PS_PLUS_COMMON_HASHER_H_

//Note: this Hasher is only used to decide
//      where a hash id's key range,
//      don't use it in hashmap

namespace ps {

class Hasher {
 public:
  static const int kTargetRange = 65536;
  static int Hash128(int64_t x_, int64_t y_) {
    uint64_t x = x_, y = y_;
    register unsigned int p = 0x0319; // MagicNumber
    register unsigned int s = 0;
    s = x & 0xFFFFL;
    p = ((p ^ s) * kP) & 0xFFFF;
    s = (x & 0xFFFF0000L) >> 16;
    p = ((p ^ s) * kP) & 0xFFFF;
    s = (x & 0xFFFF00000000L) >> 32;
    p = ((p ^ s) * kP) & 0xFFFF;
    s = (x & 0xFFFF000000000000L) >> 48;
    p = ((p ^ s) * kP) & 0xFFFF;
    s = y & 0xFFFFL;
    p = ((p ^ s) * kP) & 0xFFFF;
    s = (y & 0xFFFF0000L) >> 16;
    p = ((p ^ s) * kP) & 0xFFFF;
    s = (y & 0xFFFF00000000L) >> 32;
    p = ((p ^ s) * kP) & 0xFFFF;
    s = (y & 0xFFFF000000000000L) >> 48;
    p = ((p ^ s) * kP) & 0xFFFF;
    return p;
  }
  static uint32_t Hash64(int64_t x_) {
    int p = x_ % kTargetRange;
    return p >= 0 ? p : p + kTargetRange;
  }
 private:
  static const int kP = 397; // Prime
};

}

#endif

