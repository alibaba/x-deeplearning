/*
 * \file murmurhash.h
 * \desc murmur hash function
 */
#pragma once

#include <stdint.h>

namespace blaze {

static inline uint64_t MurmurHash64A(const char *key, uint64_t len, uint64_t seed = 0) {
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t* data = (const uint64_t *)key;
  const uint64_t* end  = data + (len / 8);

  while (data != end) {
    uint64_t k = *data++;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }
  const unsigned char * data2 = (const unsigned char*)data;

  switch (len & 7) {
    case 7: h ^= (uint64_t)((uint64_t)data2[6] << (uint64_t)48);
    case 6: h ^= (uint64_t)((uint64_t)data2[5] << (uint64_t)40);
    case 5: h ^= (uint64_t)((uint64_t)data2[4] << (uint64_t)32);
    case 4: h ^= (uint64_t)((uint64_t)data2[3] << (uint64_t)24);
    case 3: h ^= (uint64_t)((uint64_t)data2[2] << (uint64_t)16);
    case 2: h ^= (uint64_t)((uint64_t)data2[1] << (uint64_t)8 );
    case 1: h ^= (uint64_t)((uint64_t)data2[0]                );
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

}  // namespace blaze