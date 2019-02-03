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

#ifndef TDM_SERVING_BASE_FILE_FILE_MONITOR_H_
#define TDM_SERVING_BASE_FILE_FILE_MONITOR_H_

#include <string>

namespace tdm_serving {
namespace util {

struct MonitorCookie;

enum WatchEvent {
  WE_CREATE,
  WE_MODIFY,
  WE_DELETE
};

typedef void (*WatchAction)(const std::string& fileName, WatchEvent ev,
                             void* args);

struct FileMonitor {
  static MonitorCookie* Watch(const std::string& to_watch_file,
                              WatchAction action, void* args);
  static void UnWatch(const std::string& to_watch_file);
  static void UnWatch(MonitorCookie* cookie);
};

}  // namespace util
}  // namespace tdm_serving

#endif  // TDM_SERVING_BASE_FILE_FILE_MONITOR_H_
