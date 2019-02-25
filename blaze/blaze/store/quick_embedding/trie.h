/*!
 * \file trie.h
 * \desc Trie tree, string -> int16, first level index of embedding
 * \attention: Charactor of key must be common ascii 0~127
 */
#pragma once

#include <stdint.h>
#include <memory.h>
#include <string>

#include "blaze/store/quick_embedding/serializable.h"

namespace {
const int kMaxBranchNum = 128;  // common ascii size
const uint16_t kInvalid = 0xFF;
}  // namespace

namespace blaze {
namespace store {

class Trie : public Serializable {
 public:
  typedef std::string Key;
  typedef uint16_t Value;
  typedef uint8_t BranchSize;
  typedef uint8_t TrieNodeIndex;
  typedef uint32_t TrieNodeOffset;

  typedef struct Node {
    Node() : value(kInvalid), branch_num(0) {
      memset(next, 0, sizeof(Node *) * kMaxBranchNum);
    }

    Value value;
    BranchSize branch_num;
    Node *next[kMaxBranchNum];
  } TrieNode;

  Trie();

  ~Trie() override;

  // lookup value by key
  bool Lookup(const char *key, Value *value) const;

  // trie tree byte array size
  uint64_t ByteArraySize() const override;

  // dump to output stream
  bool Dump(std::ostream *os) const override;

  // load from input stream
  bool Load(std::istream *is) override;

 protected:
  // destroy trie
  void Destroy(TrieNode *root);

  // create trie node by binary data and offset
  TrieNode *GetTrieNode(const char *bytes, const TrieNodeOffset &offset) const;

 protected:
  TrieNode *root_;
  char *bytes_;
  uint64_t byte_size_;
  uint64_t capacity_size_;
};

class BulkLoadTrie : public Trie {
 public:
  BulkLoadTrie();

  ~BulkLoadTrie() override;

  // insert key/value
  bool PreInsert(const Key &key, const Value &value);

  // batch build
  bool BulkLoad();

 protected:
  // allocate new memory when expand
  bool AllocateMemory(size_t len);

  // calculate byte array size of trie node
  size_t CalcTrieNodeByteSize(TrieNode *node) const;
};

}  // namespace store
}  // namespace blaze

