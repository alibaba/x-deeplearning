/*
 * Copyright 1999-2018 Alibaba Group.
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
#ifndef XDL_CORE_LIB_ANY_H_
#define XDL_CORE_LIB_ANY_H_

#include <iostream>
#include <string>
#include <memory>
#include <typeindex>

class Any {
 public:
  Any() 
    : type_(std::type_index(typeid(void))) {}
  Any(const Any& rhs) 
    : ptr_(rhs.clone())
    , type_(rhs.type_) {}
  Any(Any&& rhs) 
    : ptr_(std::move(rhs.ptr_))
    , type_(std::move(rhs.type_)) {}

  template<class T, class = typename std::enable_if<!std::is_same<typename std::decay<T>::type, Any>::value, T>::type>
  Any(T&& value)
    : ptr_(new Holder<typename std::decay<T>::type>(std::forward<T>(value)))
    , type_(typeid(typename std::decay<T>::type)) {}

  bool Empty() {
    return ptr_.get() == nullptr;
  }

  template<class T>
  T& AnyCast() const {
    if (type_ != std::type_index(typeid(T))) {
      throw std::bad_cast();
    }

    auto ptr = dynamic_cast<Holder<T>*>(ptr_.get());
    return ptr->holder_;
  }

  Any& operator=(const Any& rhs) {
    if (ptr_ == rhs.ptr_) {
      return *this;
    }

    ptr_ = rhs.clone();
    type_ = rhs.type_;
    return *this;
  }

 private:
  class PlaceHolder;
  using PlaceHolderPtr = std::unique_ptr<PlaceHolder>;

  class PlaceHolder {
   public:
    virtual PlaceHolderPtr clone() const = 0;
    virtual ~PlaceHolder() {}
  };

  template<typename T>
  class Holder : public PlaceHolder {
   public:
    Holder(const T& arg) 
      : holder_(arg) {}

    Holder(T&& arg) 
      : holder_(std::forward<T>(arg)) {}

    PlaceHolderPtr clone() const {
      return PlaceHolderPtr(new Holder(holder_));
    }

    T holder_;
  };

  PlaceHolderPtr clone() const {
    if (ptr_) {
      return ptr_->clone();
    }

    return nullptr;
  }

  PlaceHolderPtr ptr_;
  std::type_index type_;
};

#endif
