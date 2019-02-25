/*
 * \file reshape_op.h 
 * \brief The reshape operation
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"

namespace blaze {

template <class Context>
class ReshapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  ReshapeOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) { }

  bool RunOnDevice() override {
    Blob* x = this->Input(0);
    Blob* shape_blob = this->Input(1);
    Blob* y = this->Output(0);

    std::vector<int32_t> rshape;
    for (size_t i = 0; i < shape_blob->size(); ++i) {
      rshape.push_back(shape_blob->as<int32_t>()[i]);
    }

    const std::vector<size_t>& shape = x->shape();
    std::vector<size_t> new_shape(rshape.size());
    int unknown_pos = -1;
    size_t known_size = 1;
    for (size_t i = 0; i < rshape.size(); ++i) {
      if (rshape[i] == 0) {
        new_shape[i] = shape[i]; known_size *= new_shape[i];
      } else if (rshape[i] < 0) {
        CHECK_EQ(unknown_pos, -1, "can not support two unknown dims");
        unknown_pos = i;
      } else {
        new_shape[i] = rshape[i]; known_size *= new_shape[i];
      }
    }
    if (unknown_pos >= 0) {
      new_shape[unknown_pos] = x->size() / known_size;
    } else {
      CHECK_EQ(known_size, x->size(), "known_size=", known_size, " x->size()=", x->size());
    }

    y->RefReshape(new_shape, x->as<char>());
    return true;
  }
};

}  // namespace blaze

