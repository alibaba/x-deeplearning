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

#ifndef PS_SERVICE_SEASTAR_LIB_COMMON_H_
#define PS_SERVICE_SEASTAR_LIB_COMMON_H_

namespace ps {
namespace service {
namespace seastar {

const int SEASTAR_REQUEST_PROCESSOR_ID = 1;
const int SEASTAR_RESPONSE_PROCESSOR_ID = 2;

// scheduler api id
const uint64_t SCHEDULER_REGISTERNODE_FUNC_ID = 0;
const uint64_t SCHEDULER_GETCLUSTERINFO_FUNC_ID = 1;
const uint64_t SCHEDULER_SAVE_FUNC_ID = 2;
const uint64_t SCHEDULER_RESTORE_FUNC_ID = 3;
const uint64_t SCHEDULER_UPDATEVARIABLEINFO_FUNC_ID = 4;
const uint64_t SCHEDULER_GETVARIABLEINFO_FUNC_ID = 5;

} // namespace seastar
} // namespace service
} // namespace ps

#endif // PS_SERVICE_SEASTAR_LIB_COMMON_H_
