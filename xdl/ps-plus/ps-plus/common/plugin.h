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

#ifndef PS_COMMON_REGISTER_H
#define PS_COMMON_REGISTER_H

#include <unordered_map>
#include <memory>
#include <iostream>

namespace ps {

template <typename T, typename Tidx>
class PluginManager {
public:
  T* GetPlugin(const Tidx& id) {
    auto iter = plugins_.find(id);
    if (iter == plugins_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  void RegisterPlugin(const Tidx& id, T* plugin) {
    if (!plugins_.insert({id, plugin}).second) {
      std::cerr << "Plugin Already Exist" << std::endl;
      abort();
    }
  }

  static PluginManager* GetInstance() {
    static PluginManager instance;
    return &instance;
  }
  ~PluginManager() {
    for (auto& it : plugins_) {
      delete it.second;
    }
  }
private:
  PluginManager() {}

  static std::unique_ptr<PluginManager> instance_;
  std::unordered_map<Tidx, T*> plugins_;
};

template <typename TBase, typename Tidx>
class PluginRegister {
public:
  PluginRegister(const Tidx& id, TBase* plugin) {
    PluginManager<TBase, Tidx>::GetInstance()->RegisterPlugin(id, plugin);
  };
};

template <typename TBase, typename Tidx>
TBase* GetPlugin(const Tidx& id) {
  return PluginManager<TBase, Tidx>::GetInstance()->GetPlugin(id);
}

template <typename TBase>
TBase* GetPlugin(const char* id) {
  return GetPlugin<TBase, std::string>(id);
}

}

#define PLUGIN_REGISTER_CONCAT(x, y)  x##y

#define PLUGIN_REGISTER_INTERNAL(BASETYPE, IDTYPE, REGISTER_ID, ID, PLUGINTYPE, ...) \
  static ps::PluginRegister<BASETYPE, IDTYPE> PLUGIN_REGISTER_CONCAT(PLUGIN_REGISTER_, REGISTER_ID)(ID, new PLUGINTYPE{__VA_ARGS__})

#define PLUGIN_REGISTER_TYPE(BASETYPE, IDTYPE, ID, PLUGINTYPE, ...) \
  PLUGIN_REGISTER_INTERNAL(BASETYPE, IDTYPE, __COUNTER__, ID, PLUGINTYPE, __VA_ARGS__)

#define PLUGIN_REGISTER(BASETYPE, NAME, PLUGINTYPE, ...) \
  PLUGIN_REGISTER_TYPE(BASETYPE, std::string, #NAME, PLUGINTYPE, __VA_ARGS__)

#endif
