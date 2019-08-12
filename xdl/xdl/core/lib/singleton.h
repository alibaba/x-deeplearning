/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef XDL_CORE_LIB_SINGLETON_H_
#define XDL_CORE_LIB_SINGLETON_H_

#include <utility>
#include <memory>
#include <mutex>

namespace xdl {

template<typename T>
class SingletonBase {
 public:
  friend class std::unique_ptr<T>;

  static T *Get() {
    return Instance();
  }

  /// Create singleton instance
  template<typename... Args>
  static T *Instance(Args &&... args) {
    if (instance_ == nullptr) {
      std::unique_lock<std::mutex> lock(instance_mu_);
      if (instance_ == nullptr) {
        instance_.reset(new T(std::forward<Args>(args)...));
      }
    }
    return instance_.get();
  }

 protected:
  SingletonBase() {}
  virtual ~SingletonBase() {}

 private:
  static std::mutex instance_mu_;
  static std::unique_ptr<T> instance_;
};

template<class T> std::unique_ptr<T> SingletonBase<T>::instance_;
template<class T> std::mutex SingletonBase<T>::instance_mu_;

template <typename T>
class Singleton : public SingletonBase<T>{ };

}  // namespace xdl

#endif // XDL_CORE_LIB_SINGLETON_H_
