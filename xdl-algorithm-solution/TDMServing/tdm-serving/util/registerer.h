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

// common register function for all class
// the class is required to have none parameter constructor
// Usage:
// (1) step_1: define base class
//      class BaseClass {  // base class
//          ...
//      };
// (2) step_2: define base class register
//      REGISTER_REGISTERER(BaseClass);
// (3) step_3: define sub class register
//      #define REGISTER_BASECLASS(title, name) REGISTER_CLASS(BaseClass, title, name)  // NOLINT
// (4) step_4: define sub class
//     class Sub1 : public BaseClass {
//          ...
//     };
// (5) step_5: register sub class in .cpp
//     REGISTER_BASECLASS(sub1, Sub1);
// (6) step_6: now we can use the reigster to get Instance now
//     BaseClass *base = BaseClassRegisterer::GetInstanceByTitle("sub1");

#ifndef TDM_SERVING_UTIL_REGISTERER_H_
#define TDM_SERVING_UTIL_REGISTERER_H_

#include <map>
#include <string>
#include "common/common_def.h"
#include "util/singleton.h"

namespace regist {
// idea from boost any but make it more simple and don't use type_info.
class Any {
 public:
  Any() : content_(NULL) {}

  template<typename ValueType>
  explicit Any(const ValueType &value)
      : content_(new Holder<ValueType>(value)) {}

  Any(const Any &other)
      : content_(other.content_ ? other.content_->clone() : NULL) {
  }

  ~Any() {
      delete content_;
  }

  template<typename ValueType>
  ValueType *any_cast() {
    return content_ ? &static_cast<Holder<ValueType> *>(content_)->held_ : NULL;
  }

 private:
  class PlaceHolder {
   public:
    virtual ~PlaceHolder() {}
    virtual PlaceHolder *clone() const = 0;
  };

  template<typename ValueType>
  class Holder : public PlaceHolder {
   public:
    explicit Holder(const ValueType &value) : held_(value) {}
    virtual PlaceHolder *clone() const {
      return new Holder(held_);
    }

    ValueType held_;
  };

  PlaceHolder *content_;
};

class ObjectFactory {
 public:
  ObjectFactory() {}
  virtual ~ObjectFactory() {}
  virtual Any NewInstance() {
    return Any();
  }
  virtual Any GetSingletonInstance() {
    return Any();
  }
 private:
  DISALLOW_COPY_AND_ASSIGN(ObjectFactory);
};

typedef std::map<std::string, ObjectFactory*> FactoryMap;
typedef std::map<std::string, FactoryMap> BaseClassMap;
BaseClassMap& global_factory_map();

}  // namespace regist

#define REGISTER_REGISTERER(clazz) \
class clazz ## Registerer { \
 public: \
  static clazz *GetInstanceByTitle(const ::std::string &title) { \
    ::regist::FactoryMap& map = \
        ::regist::global_factory_map()[#clazz]; \
    ::regist::FactoryMap::iterator iter = map.find(title); \
    if (iter == map.end()) { \
      return NULL; \
    } \
    ::regist::Any object = iter->second->NewInstance(); \
    return *(object.any_cast<clazz*>()); \
  } \
  static clazz* GetSingletonInstanceByTitle(const ::std::string& title) { \
    ::regist::FactoryMap& map = \
        ::regist::global_factory_map()[#clazz]; \
    ::regist::FactoryMap::iterator iter = map.find(title); \
    if (iter == map.end()) { \
      return NULL; \
    }\
    ::regist::Any object = iter->second->GetSingletonInstance(); \
    return *(object.any_cast<clazz*>()); \
  } \
  static const ::std::string GetUniqInstanceTitle() { \
    ::regist::FactoryMap &map = \
    ::regist::global_factory_map()[#clazz]; \
    assert(map.size() == 1); \
    return map.begin()->first; \
  } \
  static clazz *GetUniqInstance() { \
    ::regist::FactoryMap &map = \
        ::regist::global_factory_map()[#clazz]; \
    assert(map.size() == 1); \
    ::regist::Any object = map.begin()->second->NewInstance(); \
    return *(object.any_cast<clazz*>()); \
  } \
  static bool IsValid(const ::std::string &title) { \
    ::regist::FactoryMap &map = \
        ::regist::global_factory_map()[#clazz]; \
    return map.find(title) != map.end(); \
  } \
}; \

// title is the name of the class in conf
// name is the real subclass name
#define REGISTER_CLASS(clazz, title, name) \
namespace regist { \
class ObjectFactory##title: public ::regist::ObjectFactory { \
 public: \
  ::regist::Any NewInstance() { \
    return ::regist::Any(new name()); \
  } \
  ::regist::Any GetSingletonInstance() { \
    return ::regist::Any(&(::tdm_serving::util::Singleton<name>::Instance())); \
  } \
}; \
__attribute__((constructor)) void register_factory_##title() { \
  ::regist::FactoryMap &map = \
      ::regist::global_factory_map()[#clazz]; \
  if (map.find(#title) == map.end()) { \
    map[#title] = new ObjectFactory##title(); \
  } \
} \
}  // namespace regist

#endif  // TDM_SERVING_UTIL_REGISTERER_H_
