#include <iostream>
#include "gtest/gtest.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/common/thread_pool.h"

using ps::Hash128Key;
using ps::HashMap;
using ps::Range;
using ps::Status;
using std::vector;

TEST(HashMap64Test, Get) {
  std::unique_ptr<HashMap> hashmap(new ps::HashMapImpl<int64_t>(1));  
  int64_t keys[] = {1, 2, 3, 4};
  vector<size_t> ids;
  tbb::concurrent_vector<size_t> reused_ids;
  size_t filtered;
  int64_t max = hashmap->Get((const int64_t*)keys, 4ul, false, 1.0, &ids, &reused_ids, &filtered);
  EXPECT_EQ(4, max);
  EXPECT_EQ(4u, ids.size());
  // NOTE: the latter part of keys would get id first
  size_t total = 0;
  for (size_t i = 0; i < 4; i++) {
    total += ids[i];
  }
  EXPECT_EQ(6ul, total);
  EXPECT_EQ(0ul, reused_ids.size());

  int64_t keys1[] = {4, 3, 2, 1, 13, 14};
  max = hashmap->Get((const int64_t*)keys1, 6ul, false, 1.0, &ids, &reused_ids, &filtered);
  EXPECT_EQ(6, max);
  EXPECT_EQ(6u, ids.size());
  total = 0;
  for (size_t i = 0; i < 4; i++) {
    total += ids[i];
  }
  EXPECT_EQ(6ul, total);
  total = 0;
  for (size_t i = 4; i < 6; i++) {
    total += ids[i];
  }
  EXPECT_EQ(9u, total);  
  EXPECT_EQ(0u, reused_ids.size());
}

TEST(HashMap128Test, Get) {
  std::unique_ptr<HashMap> hashmap(new ps::HashMapImpl<Hash128Key>(1));
  int64_t keys[] = {1, 2, 3, 4};
  vector<size_t> ids;
  tbb::concurrent_vector<size_t> reused_ids;
  size_t filtered;
  int64_t max = hashmap->Get((const int64_t*)keys, 2ul, false, 1.0, &ids, &reused_ids, &filtered);
  EXPECT_EQ(2, max);
  EXPECT_EQ(2u, ids.size());
  // NOTE: the latter part of keys would get id first
  size_t total = 0;
  for (size_t i = 0; i < 2; i++) {
    total += ids[i];
  }
  EXPECT_EQ(1, total);  
  EXPECT_EQ(0u, reused_ids.size());

  int64_t keys1[] = {4, 3, 2, 1, 13, 14};
  max = hashmap->Get((const int64_t*)keys1, 3ul, false, 1.0, &ids, &reused_ids, &filtered);
  EXPECT_EQ(5, max);
  EXPECT_EQ(3u, ids.size());
  total = 0;
  for (size_t i = 0; i < 3; i++) {
    total += ids[i];
  }
  EXPECT_EQ(9, total);
  EXPECT_EQ(0u, reused_ids.size());
}

TEST(HashMap128Test, BloomFilter) {
  std::unique_ptr<HashMap> hashmap(new ps::HashMapImpl<Hash128Key>(1));
  hashmap->SetBloomFilterThrethold(2);
  int64_t keys[] = {1, 2, 3, 4};
  vector<size_t> ids;
  tbb::concurrent_vector<size_t> reused_ids;
  size_t filtered;
  int64_t max = hashmap->Get((const int64_t*)keys, 2ul, false, 1.0, &ids, &reused_ids, &filtered);
  EXPECT_EQ(max, 0);
  max = hashmap->Get((const int64_t*)keys, 2ul, false, 1.0, &ids, &reused_ids, &filtered);
  EXPECT_EQ(max, 2);  
}


TEST(HashMap128Test, Erase) {
  std::unique_ptr<HashMap> hashmap(new ps::HashMapImpl<Hash128Key>(1));
  int64_t keys[] = {1, 2};
  vector<size_t> ids;
  tbb::concurrent_vector<size_t> reused_ids;
  size_t filtered;
  int64_t max = hashmap->Get(keys, 1ul, false, 1.0, &ids, &reused_ids, &filtered);
  ASSERT_EQ(1, max);
  ASSERT_EQ(1, ids.size());
  ASSERT_EQ(0, ids[0]);
  ASSERT_EQ(0, reused_ids.size());   
  int64_t keys2[] = {3, 4};
  max = hashmap->Get(keys2, 1ul, false, 1.0, &ids, &reused_ids, &filtered);
  ASSERT_EQ(2, max);
  ASSERT_EQ(1, ids.size());
  ASSERT_EQ(1, ids[0]);
  ASSERT_EQ(0, reused_ids.size());
  int64_t del_keys[] = {3, 4};
  hashmap->Erase(del_keys, 1);
  int64_t keys3[] = {1, 2, 5, 6};
  max = hashmap->Get(keys3, 2, false, 1.0, &ids, &reused_ids, &filtered);
  ASSERT_EQ(2, max);
  EXPECT_EQ(2u, ids.size());
  EXPECT_EQ(0, ids[0]);
  // reuse id:0 
  EXPECT_EQ(1, ids[1]);
  EXPECT_EQ(1u, reused_ids.size());
  EXPECT_EQ(1, reused_ids[0]);

  int64_t del_keys1[] = {1, 2};
  hashmap->Erase(del_keys1, 1);
  int64_t keys4[] = {5, 6, 1, 2, 3, 4, 7, 8};
  max = hashmap->Get(keys4, 4, false, 1.0, &ids, &reused_ids, &filtered);
  EXPECT_EQ(4u, max);
  EXPECT_EQ(4u, ids.size());
  // <5, 6> old id:1
  EXPECT_EQ(1, ids[0]);
  size_t total = 0;
  for (size_t i = 0; i < 3; i++) {
    total += ids[i+1];
  }
  EXPECT_EQ(5, total);
}

TEST(HashMap128Test, MultiThread) {
  int thread_count = 10;
  size_t key_count = 20000l;
  std::unique_ptr<HashMap> hashmap(new ps::HashMapImpl<Hash128Key>(key_count));
  int64_t* keys = new int64_t[key_count];
  for (size_t i = 0; i < key_count; i++) {
    keys[i] = i;
  }
  std::atomic<size_t> total(0);
  auto start = std::chrono::system_clock::now();
  ps::MultiThreadDoTBB(thread_count, [&](const Range& r) {
        for (size_t i = r.begin; i < r.end; i++) {
          vector<size_t> ids;
          tbb::concurrent_vector<size_t> reused_ids;
          size_t filtered;
          hashmap->Get(keys + i* key_count/thread_count, key_count/2/thread_count, false, 1.0, &ids, &reused_ids, &filtered);
          EXPECT_EQ(key_count/2/thread_count, ids.size());
          size_t sub_total = 0;
          for (size_t j = 0; j < ids.size(); j++) {
            sub_total += ids[j];
          }
          total.fetch_add(sub_total);
        }
        return Status::Ok();
      });
  EXPECT_EQ(49995000, total);
  auto end = std::chrono::system_clock::now();
  std::cout << "insert " << key_count/2 << " keys, takes " << (end-start).count()/1000000 << "ms" <<std::endl;
  delete [] keys;
}
