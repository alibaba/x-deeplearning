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

#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/server/slice.h"

namespace ps {
namespace server {
namespace udf {

class SliceToTensor : public SimpleUdf<TensorSlices, Tensor*> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const TensorSlices& slices, Tensor* result) const {
    TensorShape new_shape;
    const Tensor* t = &slices.tensor;
    if (slices.dim_part < 0) {
      *result = slices.tensor;
    } else {
      std::vector<size_t> dims(1, slices.slice_size);
      if ((size_t)slices.dim_part > t->Shape().Size()) {
        return Status::ArgumentError("Slice dim_part Error");
      }

      dims.insert(dims.end(), 
		  t->Shape().Dims().begin() + slices.dim_part, 
		  t->Shape().Dims().end());
      new_shape = TensorShape(dims);
      new_shape.Set(0, slices.slice_id.size());
      size_t buf_size = 0;
      CASES(t->Type(), {
	  buf_size = slices.slice_id.size() * slices.slice_size * sizeof(T);
	});

      char* buf = new char[buf_size];
      char* base = t->Raw<char>();
      CASES(t->Type(), {
	  size_t chunk_size = slices.slice_size * sizeof(T);
	  for (size_t i = 0; i < slices.slice_id.size(); ++i) {
	    memcpy(buf + i * chunk_size, 
		   base + slices.slice_id[i] * chunk_size, 
		   chunk_size);
	  }
	});

      *result = ps::Tensor(t->Type(), new_shape, buf, nullptr);
    }

    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(SliceToTensor, SliceToTensor);

}
}
}

