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

#ifndef XDL_IO_DATA_POOL_H_
#define XDL_IO_DATA_POOL_H_

#include "xdl/core/lib/object_pool.h"
#include "xdl/core/lib/singleton.h"
#include "xdl/data_io/batch.h"
#include "xdl/data_io/sgroup.h"


namespace xdl {
namespace io {

class SGroupPool: public ObjectPool<SGroup>, public Singleton<SGroupPool> {
};

class BatchPool: public ObjectPool<Batch>, public Singleton<BatchPool> {
};

class CachePool: public ObjectPool<Batch>, public Singleton<BatchPool> {
};

}  // namespace io
}  // namespace xdl

#endif  // XDL_IO_DATA_POOL_H_