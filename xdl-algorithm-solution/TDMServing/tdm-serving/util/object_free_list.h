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

#ifndef TDM_SERVING_UTIL_OBJECT_FREE_LIST_H_
#define TDM_SERVING_UTIL_OBJECT_FREE_LIST_H_

#include <typeinfo>
#include "common/common_def.h"
#include "util/concurrency/object_pool.h"
#include "util/concurrency/mutex.h"
#include "util/concurrency/spin.h"

namespace tdm_serving {
namespace util {

// free object pool, for memory management 
// one for each object type
// singleton
template <typename T,
          typename Allocator = ObjectPoolNormalAllocator,
          typename LockType = AdaptiveMutex>
class ObjList {
 public:
  inline static ObjList<T, Allocator>& Instance() {
    if (nullptr == obj_list_ptr_) {
      MutexLocker lock(&ObjList::local_lock_);
      if (nullptr == obj_list_ptr_) {
        obj_list_ptr_ = new ObjList<T, Allocator, LockType>;
      }
    }

    return *obj_list_ptr_;
  }

  T* Get() {
    return obj_pool_->Acquire();
  }

  void Free(const T* p, bool is_erase = false) {
    obj_pool_->Release(p, is_erase);
  }

 private:
  ObjList() {
    obj_pool_ = new ObjectPool<T, Allocator, LockType>(ktObjectPoolInitSize);
  }

  ~ObjList() {
    obj_pool_->Clear();
    if (obj_pool_ != nullptr) {
      delete obj_pool_;
    }
    obj_pool_ = nullptr;
    delete this;
  }

  static LockType local_lock_;
  ObjectPool<T, Allocator, LockType>* obj_pool_ = nullptr;
  static ObjList<T, Allocator, LockType>* obj_list_ptr_;
};

template<typename T, typename Allocator, typename LockType>
  LockType ObjList<T, Allocator, LockType>::local_lock_;
template<typename T, typename Allocator, typename LockType>
  ObjList<T, Allocator, LockType>*
  ObjList<T, Allocator, LockType>::obj_list_ptr_ = nullptr;

#define GET_INSTANCE(C) \
    (::tdm_serving::util::ObjList<C>::Instance().Get())

#define FREE_INSTANCE(C, val) \
    ::tdm_serving::util::ObjList<C>::Instance().Free(static_cast<const C*>(val))

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_UTIL_OBJECT_FREE_LIST_H_

