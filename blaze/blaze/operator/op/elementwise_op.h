/*
 * \file elementwise_op.h
 * \desc The elementwise operator which support Broadcasting.
 */
#pragma once

#include "blaze/operator/operator.h"
#include "blaze/common/exception.h"
#include "blaze/common/types.h"
#include "blaze/math/broadcast.h"
#include "blaze/common/log.h"

namespace blaze {

template <typename DType>
struct ElementwiseParam {
  bool* condition;
  size_t condition_n;
  std::vector<TIndex> condition_shape;
  DType* x;
  size_t x_n;
  std::vector<TIndex> x_shape;
  DType* y;
  size_t y_n;
  std::vector<TIndex> y_shape;
  DType* z;
  size_t z_n;
  std::vector<TIndex> z_shape;

  ElementwiseParam(DType* x, size_t x_n, const std::vector<TIndex>& x_shape,
      DType* y, size_t y_n, const std::vector<TIndex>& y_shape,
      DType* z, size_t z_n, const std::vector<TIndex>& z_shape) :
      x(x), x_n(x_n), x_shape(x_shape),
      y(y), y_n(y_n), y_shape(y_shape),
      z(z), z_n(z_n), z_shape(z_shape) { }
};

template <typename IType, typename DType>
struct TernaryElementwiseParam {
  IType* condition;
  size_t condition_n;
  std::vector<TIndex> condition_shape;
  DType* x;
  size_t x_n;
  std::vector<TIndex> x_shape;
  DType* y;
  size_t y_n;
  std::vector<TIndex> y_shape;
  DType* z;
  size_t z_n;
  std::vector<TIndex> z_shape;

  TernaryElementwiseParam(IType* condition, size_t conditon_n, const std::vector<TIndex>& condition_shape,
                          DType* x, size_t x_n, const std::vector<TIndex>& x_shape,
                          DType* y, size_t y_n, const std::vector<TIndex>& y_shape,
                          DType* z, size_t z_n, const std::vector<TIndex>& z_shape) :
      condition(condition), condition_n(condition_n), condition_shape(condition_shape),
      x(x), x_n(x_n), x_shape(x_shape),
      y(y), y_n(y_n), y_shape(y_shape),
      z(z), z_n(z_n), z_shape(z_shape) { }
};

template <class Context>
class ElementwiseOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);
  
  ElementwiseOp(const OperatorDef& def, Workspace* workspace) :
      Operator<Context>(def, workspace) { }

 protected:
   void CheckBroadcast(const std::vector<TIndex>& a_shape,
                       const std::vector<TIndex>& b_shape) {
    int a_idx = 0;
    int b_idx = 0;
    if (a_shape.size() >= b_shape.size()) {
      a_idx += (a_shape.size() - b_shape.size());
    } else {
      b_idx += (b_shape.size() - a_shape.size());
    }
    for (; a_idx < a_shape.size() && b_idx < b_shape.size(); ++a_idx, ++b_idx) {
      BLAZE_CONDITION_THROW(a_shape[a_idx] == b_shape[b_idx] ||
                            a_shape[a_idx] == 1 || b_shape[b_idx] == 1,
                            "a->shape()[", a_idx,
                            "]=", a_shape[a_idx],
                            "; b->shape()[", b_idx,
                            "]=", b_shape[b_idx],
                            " not equal or meet broadcast rule");
    }
  }

  virtual void CheckValid() {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);

    BLAZE_CONDITION_THROW(a->data_type() == b->data_type(),
                          "a->data_type()=",
                          a->data_type(),
                          " b->data_type()=",
                          b->data_type(), " op=", this->def_.DebugString());

    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();
    CheckBroadcast(a_shape, b_shape);
  }

  virtual void Reshape() {
    Blob* a = this->Input(0);
    Blob* b = this->Input(1);
    Blob* c = this->Output(0);

    // apply the Reshap func of broadcast 
    std::vector<TIndex> new_shape;
    MBroadcasting::BroadcastShape(a->shape(), b->shape(), new_shape);
    c->Reshape(new_shape); 
  }

  bool NeedBroadcast() {
    Blob* a = this->Input(0);
    Blob* b = this->InputSize() < 2 ? this->Output(0) : this->Input(1);
    auto a_shape = a->shape();
    auto b_shape = b->shape();
    if (a_shape.size() == b_shape.size()) {
      for (int i = 0; i < a_shape.size(); ++i) {
        if (a_shape[i] != b_shape[i]) {
          return true; 
        } 
      }
      return false;
    }
    return true;
  }   
};

template <class Context>
class ElementwiseAddOp final : public ElementwiseOp<Context> {
 public:
  ElementwiseAddOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

template <class Context>
class ElementwiseSubOp final : public ElementwiseOp<Context> {
 public:
  ElementwiseSubOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

template <class Context>
class ElementwiseMulOp final : public ElementwiseOp<Context> {
 public:
  ElementwiseMulOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

template <class Context>
class ElementwiseDivOp final : public ElementwiseOp<Context> {
 public:
  ElementwiseDivOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

template <class Context>
class ElementwiseEqualOp final : public ElementwiseOp<Context> {
 public:
  ElementwiseEqualOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

template <class Context>
class ElementwiseNotEqualOp final : public ElementwiseOp<Context> {
 public:
  ElementwiseNotEqualOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

template <class Context>
class ElementwiseMaxOp final : public ElementwiseOp<Context> {
 public:
  ElementwiseMaxOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

template <class Context>
class ElementwiseMinOp final : public ElementwiseOp<Context> {
 public:
  ElementwiseMinOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;
};

template <class Context>
class BroadcastToOp : public ElementwiseOp<Context> {
 public:
  BroadcastToOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) {
    broadcast_shape_ = OperatorBase::GetRepeatedArgument<int>("shape"); 
  }

  virtual void CheckValid() override { }
  virtual void Reshape() override {
    Blob* a = this->Input(0);
    Blob* y = this->Output(0);
    if (this->InputSize() > 1) {
      Blob* b = this->Input(1);
      y->Reshape(b->shape());
      return;
    }
    BLAZE_CONDITION_THROW(a->shape().size() == broadcast_shape_.size(),
                          "a->shape().size()=", a->shape().size(),
                          " broadcast_shape_.size()=", broadcast_shape_.size());
    std::vector<TIndex> shape = a->shape();
    for (auto i = 0; i < broadcast_shape_.size(); ++i) {
      if (broadcast_shape_[i] > 0) {
        shape[i] = broadcast_shape_[i];
      } else if (broadcast_shape_[i] < 0) {
        BLAZE_THROW("can not support dim < 0");
      }
    }
    y->Reshape(shape);
  }

  bool RunOnDevice() override;

 protected:
  std::vector<int> broadcast_shape_;
};

template <class Context>
class WhereOp : public ElementwiseOp<Context> {
 public:
  WhereOp(const OperatorDef& def, Workspace* workspace) :
      ElementwiseOp<Context>(def, workspace) { }

  bool RunOnDevice() override;

 protected:
  virtual void CheckValid() override {
    // check data type
    Blob* condition = this->Input(0);
    Blob* x = this->Input(1);
    Blob* y = this->Input(2);
    BLAZE_CONDITION_THROW(x->data_type() == y->data_type(),
                          "x->data_type()=",
                          x->data_type(),
                          " y->data_type()=",
                          y->data_type(), " op=", this->def_.DebugString());
    // check size
    for (int i = 0; i < this->InputSize() - 1; ++i) {
      for (int j = i + 1; j < this->InputSize(); ++j) {
        const auto& a_shape = this->Input(i)->shape();
        const auto& b_shape = this->Input(j)->shape();
        this->CheckBroadcast(a_shape, b_shape);
      }
    }
  }

  virtual void Reshape() override {
    Blob* condition = this->Input(0);
    Blob* x = this->Input(1);
    Blob* y = this->Input(2);
    Blob* z = this->Output(0);

    // apply the Reshap func of broadcast
    std::vector<TIndex> tmp_shape, new_shape;
    MBroadcasting::BroadcastShape(condition->shape(), x->shape(), tmp_shape);
    MBroadcasting::BroadcastShape(tmp_shape, y->shape(), new_shape);
    z->Reshape(new_shape);
  }

  bool NeedBroadcast() {
    Blob* condition = this->Input(0);
    Blob* x = this->Input(1);
    Blob* y = this->Input(2);
    const auto& condition_shape = condition->shape();
    const auto& x_shape = x->shape();
    const auto& y_shape = y->shape();
    if (x_shape.size() == y_shape.size()
        && x_shape.size() == condition_shape.size()) {
      for (int i = 0; i < x_shape.size(); ++i) {
        if (x_shape[i] != y_shape[i] || x_shape[i] != condition_shape[i]) {
          return true;
        }
      }
      return false;
    }
    return true;
  }
};

}  // namespace blaze

