/*
 * \file store.h
 * \brief The store backend interface
 */
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace blaze {
namespace store {

enum Status {
  kOK = 0,
  kTimeout,
  kFail,
  // ...
};

enum UDFType {
  kSum = 0,
  kAvg,
  kAssign,
};

enum KeyType {
  kInt64 = 10,
};

enum ValueType {
  kFloat = 1,
  kFloat16 = 12,
};

struct Query {
  std::string table_name;  // table name
  void* keys;              // ids in a group
  void* values;            // values in a group
  int* key_seg;            // segments list
  int seg_num;             // segments num
  KeyType key_type;        // key type
  ValueType value_type;    // value type
  UDFType udf_type;        // udf type
};

struct Response {
  void* values;            // managed by storage engine
  size_t size;             // value size
  ValueType value_type;    // value type
};

typedef std::function<void(Status ret, const std::vector<Response>&, void* args)> AsynCallbackFunction;

class Store {
 public:
  virtual ~Store() {}

  virtual bool Load(const std::string& uri) = 0;

  virtual bool MGet(const std::vector<Query>& query,
                    std::vector<Response>* response) = 0;

  virtual bool Asyn_MGet(const std::vector<Query>& query,
                         AsynCallbackFunction cb,
                         void* args) = 0;
};

using FCreateStore=std::function<Store*(void)>;

// Store creation register
struct StoreCreationRegisterer {
  static StoreCreationRegisterer* Get() {
    static std::shared_ptr<StoreCreationRegisterer> inst(new StoreCreationRegisterer());
    return inst.get();
  }
  bool Register(const std::string& name, FCreateStore fcs);
  Store* CreateStore(const std::string& name);

 protected:
  StoreCreationRegisterer() { }
  StoreCreationRegisterer(const StoreCreationRegisterer&) = delete;
  StoreCreationRegisterer& operator=(const StoreCreationRegisterer&) = delete;

  std::unordered_map<std::string, FCreateStore> fcs_map_;
};

#define REGISTER_STORE_CREATION(name, fcs) \
    static bool status = StoreCreationRegisterer::Get()->Register(name, fcs);

}  // namespace store
}  // namespace blaze
