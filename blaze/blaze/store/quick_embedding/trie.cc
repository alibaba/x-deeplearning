/*!
 * \file trie.cc
 * \desc Trie tree, string -> int16, first level index of embedding
 * \attention: Charactor of key must be common ascii 0~127
 */

#include <memory.h>
#include <queue>

#include "blaze/store/quick_embedding/trie.h"
#include "blaze/common/log.h"

namespace {
const size_t kBaseCapacity = 8192;
const float kExpandRatio = 1.5;
}  // namespace

namespace blaze {
namespace store {

Trie::Trie() :
    root_(nullptr),
    bytes_(nullptr),
    byte_size_(0),
    capacity_size_(0) {
}

Trie::~Trie() {
  Destroy(root_);
  if (bytes_) {
    free(bytes_);
  }
}

void Trie::Destroy(TrieNode *root) {
  if (!root) {
    return;
  }
  for (int i = 0; i < kMaxBranchNum; i++) {
    Destroy(root->next[i]);
  }
  delete root;
  root = nullptr;
}

bool Trie::Lookup(const char *key, Value *value) const {
  if (!root_) {
    LOG_ERROR("empty trie tree");
    return false;
  }
  TrieNode *traverse_node = root_;
  size_t i = 0;
  size_t len = strlen(key);
  while (traverse_node && i != len) {
    int index = key[i];
    if (index < 0 || index >= kMaxBranchNum) {
      LOG_ERROR("key has invalid charactor: %s", key);
      return false;
    }
    traverse_node = traverse_node->next[index];
    i++;
  }
  if (!traverse_node)
    return false;

  *value = traverse_node->value;
  return true;
}

Trie::TrieNode *Trie::GetTrieNode(const char *bytes, const TrieNodeOffset &offset) const {
  TrieNode *node = new TrieNode();
  size_t pos = offset;
  node->value = *(Value *) (bytes + pos);
  pos += sizeof(Value);
  node->branch_num = *(BranchSize *) (bytes + pos);
  pos += sizeof(BranchSize);
  for (int i = 0; i < node->branch_num; ++i) {
    TrieNodeIndex index = *(TrieNodeIndex *) (bytes + pos);
    pos += sizeof(TrieNodeIndex);
    TrieNodeOffset offset = *(TrieNodeOffset *) (bytes + pos);
    pos += sizeof(TrieNodeOffset);
    node->next[index] = GetTrieNode(bytes, offset);
  }
  return node;
}

uint64_t Trie::ByteArraySize() const {
  return sizeof(byte_size_) + byte_size_;
}

bool Trie::Dump(std::ostream *os) const {
  if (byte_size_ == 0 || bytes_ == nullptr)
    return false;

  // [step1]: write byte size
  os->write((char *) &byte_size_, sizeof(byte_size_));
  if (!os->good()) return false;
  // [step2]: write bytes
  os->write(bytes_, byte_size_);
  if (!os->good()) return false;

  return true;
}

bool Trie::Load(std::istream *is) {
  // [step1]: load byte size
  is->read((char *) &byte_size_, sizeof(byte_size_));
  if (!is->good()) return false;
  capacity_size_ = byte_size_;
  // [step2]: reallocate memory
  char *new_bytes = reinterpret_cast<char *>(malloc(byte_size_));
  if (!new_bytes) {
    LOG_ERROR("bad allocate memory of trie while loading!");
    return false;
  }
  bytes_ = new_bytes;
  // [step3]: load bytes
  is->read(bytes_, byte_size_);
  if (!is->good()) return false;
  // refactor trie tree
  Destroy(root_);
  root_ = GetTrieNode(bytes_, 0);
  return true;
}

BulkLoadTrie::BulkLoadTrie() : Trie() {}

BulkLoadTrie::~BulkLoadTrie() {}

bool BulkLoadTrie::PreInsert(const Key &key, const Value &value) {
  // create new tree
  if (!root_)
    root_ = new TrieNode();
  TrieNode *traverse_node = root_;
  // traverse
  const char *cstr = key.c_str();
  for (int i = 0; i < strlen(cstr); ++i) {
    int index = cstr[i];
    if (index < 0 || index >= kMaxBranchNum) {
      LOG_ERROR("key has invalid charactor: %s", key.c_str());
      return false;
    }
    // charactor not exist, create new node
    if (!traverse_node->next[index]) {
      traverse_node->next[index] = new TrieNode();
      traverse_node->branch_num++;
    }
    traverse_node = traverse_node->next[index];
  }
  if (traverse_node->value != kInvalid) {  // duplicate value
    if (traverse_node->value != value) {
      LOG_ERROR("conflict value with same key=%s pre_value=%d cur_value=%d",
                key.c_str(), traverse_node->value, value);
      return false;
    }
  } else {
    traverse_node->value = value;
  }
  return true;
}

bool BulkLoadTrie::BulkLoad() {
  // bfs queue
  byte_size_ = 0;
  size_t pos = 0;
  std::queue < TrieNode * > bfs_queue;
  bfs_queue.push(root_);
  // allocate root memory
  size_t len = CalcTrieNodeByteSize(root_);
  if (!AllocateMemory(len)) {
    LOG_ERROR("allocate memory for trie root failed!");
    return false;
  }
  byte_size_ += len;
  // bfs traverse
  while (!bfs_queue.empty()) {
    TrieNode *node = bfs_queue.front();
    bfs_queue.pop();
    *(Value * )(bytes_ + pos) = node->value;
    pos += sizeof(Value);
    *(BranchSize * )(bytes_ + pos) = node->branch_num;
    pos += sizeof(BranchSize);
    for (TrieNodeIndex index = 0; index < kMaxBranchNum; ++index) {
      if (node->next[index]) {
        *(TrieNodeIndex * )(bytes_ + pos) = index;
        pos += sizeof(TrieNodeIndex);
        *(TrieNodeOffset * )(bytes_ + pos) = byte_size_;
        pos += sizeof(TrieNodeOffset);
        // allocate memory
        size_t len = CalcTrieNodeByteSize(node->next[index]);
        if (!AllocateMemory(len)) {
          LOG_ERROR("allocate memory for trie node faild!");
          return false;
        }
        byte_size_ += len;
        // push back leaf node
        bfs_queue.push(node->next[index]);
      }
    }
  }
  return true;
}

bool BulkLoadTrie::AllocateMemory(size_t len) {
  // memory enough
  if (byte_size_ + len <= capacity_size_)
    return true;
  // calc capacity size
  while (byte_size_ + len > capacity_size_) {
    capacity_size_ = static_cast<uint64_t>(capacity_size_ * kExpandRatio + kBaseCapacity);
  }
  // allocate memory
  char *new_bytes = reinterpret_cast<char *>(realloc(bytes_, capacity_size_));
  if (!new_bytes) {
    return false;
  }
  bytes_ = new_bytes;
  return true;
}

size_t BulkLoadTrie::CalcTrieNodeByteSize(TrieNode *node) const {
  size_t ret = 0;
  if (!node)
    return ret;
  // + value size
  ret += sizeof(node->value);
  // + branch length
  ret += sizeof(node->branch_num);
  // statistics total branch size
  uint32_t branch_count = 0;
  for (int i = 0; i < kMaxBranchNum; ++i) {
    if (node->next[i]) {
      branch_count++;
    }
  }
  ret += (sizeof(TrieNodeIndex) + sizeof(TrieNodeOffset)) * branch_count;
  return ret;
}

}  // namespace store
}  // namespace blaze

