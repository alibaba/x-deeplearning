/*!
 * \file hashtable.h
 * \desc Hashtable, second level index of embedding
 * \attention Key and value must be integer.
 */
#pragma once

#include "blaze/store/quick_embedding/serializable.h"

#include <vector>

namespace blaze {
namespace store {

class HashTable : public Serializable {
 public:
  typedef uint64_t Key;
  typedef uint64_t Value;

  HashTable();

  ~HashTable() override;

  // lookup value by key
  bool Lookup(const Key &key, Value *value) const;

  // hash table byte array size
  uint64_t ByteArraySize() const override;

  // dump
  bool Dump(std::ostream *os) const override;

  // load
  bool Load(std::istream *is) override;

  // get avg length
  float AvgLength() const;

 protected:
  // interior bucket node definition
  struct Bucket {
    // maximum 2^40 = 1024GB
    uint64_t offset : 40;
    // maximim size: 2^24bit
    uint32_t size : 24;
  };

  // bucket size
  uint32_t bucket_size_;
  // buckets
  Bucket *bucket_;
  // byte size
  size_t byte_size_;
  // byte array
  char *bytes_;
};

// batch build HashTable.
class BulkLoadHashTable : public HashTable {
 public:
  BulkLoadHashTable();

  ~BulkLoadHashTable() override;

  // insert key/value
  bool PreInsert(const Key &key, const Value &value);

  // batch build
  bool BulkLoad();

 protected:
  // init bucket
  bool InitBucket();

  // init node
  bool InitByte();

  struct Item {
    Key key;
    Value value;
    uint32_t bucket_id;
  };

  // compare func
  static bool Compare(const Item &item1, const Item &item2) {
    if (item1.bucket_id < item2.bucket_id)
      return true;
    else if (item1.bucket_id > item2.bucket_id)
      return false;
    else
      return item1.key < item2.key;
  }

  std::vector<Item> items_;
};

}  // namespace store
}  // namespace blaze

