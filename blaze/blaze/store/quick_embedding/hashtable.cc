/*!
 * \file hashtable.cc
 * \desc Hashtable, the second level index of embedding
 */

#include "blaze/store/quick_embedding/hashtable.h"

#include <memory.h>
#include <algorithm>

namespace {
const unsigned int kHashTableStepSize[] = {
    53, 97, 193, 389, 769,
    1543, 3079, 6151, 12289, 24593,
    49157, 98317, 196613, 393241, 786433,
    1572869, 3145739, 6291469, 12582917, 25165843,
    50331653, 100663319, 201326611, 402653189, 805306457
};
}  // namespace

namespace blaze {
namespace store {

uint32_t CalculateBucketSize(uint32_t node_size) {
  uint32_t max_step = sizeof(kHashTableStepSize) / sizeof(kHashTableStepSize[0]);
  for (uint32_t i = 0; i < max_step; ++i) {
    if (kHashTableStepSize[i] > node_size) {
      return kHashTableStepSize[i];
    }
  }
  return kHashTableStepSize[max_step - 1];
}

HashTable::HashTable() :
    bucket_size_(0),
    bucket_(nullptr),
    byte_size_(0),
    bytes_(nullptr) {}

HashTable::~HashTable() {
  if (bucket_) free(bucket_);
  if (bytes_) free(bytes_);
}

bool HashTable::Lookup(const Key &key, Value *value) const {
  if (bucket_ == nullptr || bytes_ == nullptr)
    return false;
  // [STEP1]: compute hash_code and get bucket
  uint32_t hash_code = key;
  const Bucket &bucket = bucket_[hash_code % bucket_size_];
  if (!bucket.size) return false;
  // [STEP2]: if bucket has data, scan on array
  char *ptr = bytes_ + bucket.offset;
  for (uint32_t i = 0; i < bucket.size; ++i) {
    const Key &parse_key = *(Key *) ptr;
    ptr += sizeof(Key);
    if (key == parse_key) {
      *value = *(Value *) ptr;
      return true;
    }
    ptr += sizeof(Value);
  }
  return false;
}

uint64_t HashTable::ByteArraySize() const {
  return sizeof(bucket_size_) +
         sizeof(Bucket) * bucket_size_ +
         sizeof(byte_size_) +
         byte_size_;
}

bool HashTable::Dump(std::ostream *os) const {
  if (bucket_ == nullptr || bytes_ == nullptr) {
    return false;
  }
  // [STEP1]: write bucket size
  os->write((char *) &bucket_size_, sizeof(bucket_size_));
  if (!os->good()) return false;

  // [STEP2]: write bucket
  os->write((char *) bucket_, bucket_size_ * sizeof(Bucket));
  if (!os->good()) return false;

  // [STEP3]: write byte size
  os->write((char *) &byte_size_, sizeof(byte_size_));
  if (!os->good()) return false;

  // [STEP4]: write bytes
  os->write((char *) bytes_, byte_size_);
  if (!os->good()) return false;

  return true;
}

bool HashTable::Load(std::istream *is) {
  // [STEP1]: load bucket size
  is->read((char *) &bucket_size_, sizeof(bucket_size_));
  if (!is->good()) return false;

  // [STEP2]: load bucket
  bucket_ = (Bucket *) realloc(bucket_, sizeof(Bucket) * bucket_size_);
  if (!bucket_) return false;
  is->read((char *) bucket_, sizeof(Bucket) * bucket_size_);
  if (!is->good()) return false;

  // [STEP3]: load byte size
  is->read((char *) &byte_size_, sizeof(byte_size_));
  if (!is->good()) return false;

  // [STEP4]: load bytes
  bytes_ = (char *) realloc(bytes_, byte_size_);
  if (!bytes_) return false;
  is->read((char *) bytes_, byte_size_);
  if (!is->good()) return false;

  return true;
}

float HashTable::AvgLength() const {
  size_t count = 0;
  float len = 0;
  for (uint32_t i = 0; i < bucket_size_; ++i) {
    if (bucket_[i].size == 0) continue;
    len += bucket_[i].size;
    ++count;
  }
  if (count != 0) return (float) len / count;
  return 0;
}

BulkLoadHashTable::BulkLoadHashTable() : HashTable() {}

BulkLoadHashTable::~BulkLoadHashTable() {
}

bool BulkLoadHashTable::PreInsert(const Key &key, const Value &value) {
  uint32_t hashcode = key;
  Item item = {key, value, hashcode};
  items_.push_back(item);
  return true;
}

bool BulkLoadHashTable::BulkLoad() {
  // [STEP1]: init bucket
  if (!InitBucket()) return false;

  // [STEP2]: sort by bucket_id key
  for (auto it = items_.begin(); it != items_.end(); ++it) {
    it->bucket_id = it->bucket_id % bucket_size_;
  }
  std::sort(items_.begin(), items_.end(), Compare);

  // [STEP3]: init node
  if (!InitByte()) return false;
  return true;
}

bool BulkLoadHashTable::InitBucket() {
  bucket_size_ = CalculateBucketSize(items_.size());
  bucket_ = (Bucket *) realloc(bucket_, sizeof(Bucket) * bucket_size_);
  if (!bucket_) return false;
  memset(bucket_, 0, sizeof(Bucket) * bucket_size_);
  return true;
}

bool BulkLoadHashTable::InitByte() {
  // [STEP1]: calc memory usage
  byte_size_ = (sizeof(Key) + sizeof(Value)) * items_.size();
  // [STEP2]: allocate memory
  char *new_bytes = (char *) realloc(bytes_, byte_size_);
  if (!new_bytes) return false;
  bytes_ = new_bytes;
  // [STEP3]: assignment
  char *cur = bytes_;
  for (auto it = items_.begin(); it != items_.end(); ++it) {
    // first element
    if (bucket_[it->bucket_id].size == 0) {
      bucket_[it->bucket_id].offset = cur - bytes_;
    }
    *(Key *) cur = it->key;
    cur += sizeof(Key);
    *(Value *) cur = it->value;
    cur += sizeof(Value);
    bucket_[it->bucket_id].size++;
  }
  return true;
}

}  // namespace store
}  // namespace blaze
