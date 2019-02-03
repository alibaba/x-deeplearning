/*
 * \file registry.h
 * \desc A simple registry for object creator.
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "blaze/common/common_defines.h"
#include "blaze/common/exception.h"
#include "blaze/common/log.h"
#include "blaze/common/typeid.h"

namespace blaze {

/**
 * A template class for register classes by keys, the key is the string.
 */
template <class ObjectPtrType, class... Args>
class Registry {
 public:
  typedef std::function<ObjectPtrType(Args...)> Creator;

  Registry() : registry_() { }
  void Register(const std::string& key, Creator creator) {
    std::lock_guard<std::mutex> lock(register_mutex_);
    if (registry_.count(key) != 0) {
      BLAZE_THROW("key ", key, " is duplicate registered");
    }
    registry_[key] = creator;
  }
  void Register(const std::string& key, Creator creator, const std::string& help_msg) {
    Register(key, creator);
    help_msg_[key] = help_msg;
  }
  inline bool Has(const std::string& key) const { return registry_.count(key) != 0; }

  ObjectPtrType Create(const std::string& key, Args... args) {
    if (!Has(key)) {
      // Returns nullptr if the key is not registered.
      LOG_ERROR("key=%s not exist", key.c_str());
      return nullptr;
    }
    return registry_[key](args...);
  }
  std::vector<std::string> Keys() {
    std::vector<std::string> keys;
    for (const auto& it : registry_) keys.push_back(it.first);
    return keys;
  }
  const std::map<std::string, std::string>& help_msg() const { return help_msg_; }
  const char* help_msg(const std::string& key) const {
    auto it = help_msg_.find(key);
    if (it == help_msg_.end()) return nullptr;
    return it->second.c_str();
  }

 protected:
  std::mutex register_mutex_;
  std::map<std::string, Creator> registry_;
  std::map<std::string, std::string> help_msg_;

  DISABLE_COPY_AND_ASSIGN(Registry);
};

template <class ObjectPtrType, class... Args>
class Registerer {
 public:
  Registerer(const std::string& key,
             Registry<ObjectPtrType, Args...>* registry,
             typename Registry<ObjectPtrType, Args...>::Creator creator,
             const std::string& help_msg = "") {
    registry->Register(key, creator, help_msg);
  }

  template <class DerivedType>
  static ObjectPtrType DefaultCreator(Args... args) {
    return ObjectPtrType(new DerivedType(args...));
  }
};

#define DECLARE_TYPED_REGISTRY(RegistryName, ObjectType, PtrType, ...)            \
    Registry<PtrType<ObjectType>, ##__VA_ARGS__>* RegistryName();                 \
    typedef Registerer<PtrType<ObjectType>, ##__VA_ARGS__>                        \
       Registerer##RegistryName;

#define DEFINE_TYPED_REGISTRY(RegistryName, ObjectType, PtrType, ...)             \
    Registry<PtrType<ObjectType>, ##__VA_ARGS__>* RegistryName() {                \
      static Registry<PtrType<ObjectType>, ##__VA_ARGS__>* registry =             \
        new Registry<PtrType<ObjectType>, ##__VA_ARGS__>();                       \
      return registry;                                                            \
    }

#define REGISTER_TYPED_CREATOR(RegistryName, key, ...)                            \
    namespace {                                                                   \
      static Registerer##RegistryName ANONYMOUS_VARIABLE(g_##RegistryName)(       \
         key, RegistryName(), __VA_ARGS__);                                       \
    }

#define REGISTER_TYPED_CLASS(RegistryName, key, ...)                              \
    namespace {                                                                   \
      static Registerer##RegistryName ANONYMOUS_VARIABLE(g_##RegistryName)(       \
         key,                                                                     \
         RegistryName(),                                                          \
         Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                   \
         DemangleType<__VA_ARGS__>());                                            \
    }

#define DECLARE_REGISTRY(RegistryName, ObjectType, ...)                           \
    DECLARE_TYPED_REGISTRY(                                                       \
      RegistryName, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define DEFINE_REGISTRY(RegistryName, ObjectType, ...)                            \
    DEFINE_TYPED_REGISTRY(                                                        \
      RegistryName, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define DECLARE_SHARED_REGISTRY(RegistryName, ObjectType, ...)                    \
    DECLARE_TYPED_REGISTRY(                                                       \
      RegisterName, ObjectType, std::shared_ptr, ##__VA_ARGS__)

#define DEFINE_SHARED_REGISTRY(RegistryName, ObjectType, ...)                     \
    DEFINE_TYPED_REGISTRY(                                                        \
      RegisterName, ObjectType, std::shared_ptr, ##__VA_ARGS__)

#define REGISTER_CREATOR(RegistryName, key, ...)                                  \
    REGISTER_TYPED_CREATOR(RegistryName, #key, __VA_ARGS__)

// key is the name, the second is derived class
#define REGISTER_CLASS(RegistryName, key, ...)                                    \
    REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)

}  // namespace blaze

