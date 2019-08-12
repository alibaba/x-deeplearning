/* Copyright 2018 Alibaba Group. All Rights Reserved.

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

#include <vector>
#include "tbb/concurrent_queue.h"
#include "xdl/core/framework/tensor.h"

namespace xdl {

class TBBConcurrentQueue {
public:
  TBBConcurrentQueue() {finished = false;}
  ~TBBConcurrentQueue() {};

  void SetFinished() {finished = true;}
  bool Finished() {return finished;}
  tbb::concurrent_bounded_queue<std::vector<Tensor>>* Raw() {return &queue;}
  static TBBConcurrentQueue* Global() {
    static TBBConcurrentQueue queue;
    return &queue;
  }
  
private:
  tbb::concurrent_bounded_queue<std::vector<Tensor>> queue;  
  bool finished;
};

} // namespace xdl


