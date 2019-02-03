/*
 * \file sparse_puller.h 
 * \brief The sparse puller, which pull parameter of sparse ids.
 */
#pragma once

#include <stdio.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "blaze/store/store.h"

namespace blaze {
namespace store {

// The sparse feature block format 
// feature_group:
//      name0
//
// key:
// 0)   id0_0, id0_1, id0_2
// 1)   id0_3, id0_4
//
// key_num:
// 0)   3
// 1)   2
//
// key_num_size: 2
//
// value:
// 0)   1.0, 2.0, 2.1
// 1)   1.2, 1.2
enum TruncDirection {
  kOrder = 0,
  kReverse,
};

struct SparsePullerInput {
  // version
  std::string version;
  // table name
  std::string name;
  // The sparse id 
  void* key;
  void* key_num;
  size_t key_num_size;
  // The sparse id's value
  void* value;
  int key_type;
  int value_type;
  int num_type;
  // The dim of id
  int dim;

  struct Param {
    UDFType udf_type;
    TruncDirection trunc_direction;
    int trunc_num;
  };
  std::vector<Param> in_item;
};

struct SparsePullerOutput {
  struct OutItem {
    // The out type is the same as input value type.
    void* out;
    size_t stride;
  };
  std::vector<OutItem> out_item;
};

// The sparse puller 
class SparsePuller {
 public:
  virtual ~SparsePuller() { }
  virtual Status Load(const std::string& url) { return kFail; }
  virtual Status Get(const std::vector<SparsePullerInput>& input,
                     std::vector<SparsePullerOutput>& output) { return kFail; }
};

using FCreateSparsePuller=std::function<SparsePuller*(void)>;

// Sparse puller creation register
struct SparsePullerCreationRegisterer {
  static SparsePullerCreationRegisterer* Get() {
    static std::shared_ptr<SparsePullerCreationRegisterer> inst(new SparsePullerCreationRegisterer());
    return inst.get();
  }
  bool Register(const std::string& name, FCreateSparsePuller fcs);
  SparsePuller* CreateSparsePuller(const std::string& name);

 protected:
  SparsePullerCreationRegisterer() { }
  SparsePullerCreationRegisterer(const SparsePullerCreationRegisterer&) = delete;
  SparsePullerCreationRegisterer& operator=(const SparsePullerCreationRegisterer&) = delete;

  std::unordered_map<std::string, FCreateSparsePuller> fcs_map_;
};

#define REGISTER_SPARSE_PULLER_CREATION(name, fcs) \
    static bool status = SparsePullerCreationRegisterer::Get()->Register(name, fcs);

}  // namespace store
}  // namespace blaze
