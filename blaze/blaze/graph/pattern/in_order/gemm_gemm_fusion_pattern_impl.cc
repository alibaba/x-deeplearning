/*
 * \file gemm_gemm_fusion_pattern_impl.cc
 * \brief The gemm gemm fusion pattern impl.
 */
#include "blaze/graph/pattern/in_order/gemm_gemm_fusion_pattern_impl.h"

#include "blaze/common/exception.h"

#include "blaze/common/blob.h"
#include "blaze/math/broadcast.h"
#include "blaze/math/gemm.h"

namespace blaze {

bool GemmGemmFusionPatternImpl::Match(const std::vector<ArgumentHelper*>& args,
                                      const std::vector<Node*>& nodes,
                                      Graph* graph) {
  // All the weight and bias node must be ConstantFill
  Node *node0 = nodes[0], *node1 = nodes[1];
  
  bool transA = args[1]->GetSingleArgument<bool>("transA", false);
  if (transA) return false;

  int w0_idx = node0->GetParentIdx(node0->op.input(1));
  int w1_idx = node1->GetParentIdx(node1->op.input(1));
  BLAZE_CONDITION_THROW(w0_idx >= 0 && w1_idx >= 0,
                        "w0_idx=", w0_idx, " w1_idx=", w1_idx, " ", node1->op.input(1));

  BLAZE_CONDITION_THROW(graph->node(w0_idx).op.type() == "ConstantFill",
                        "graph->node(w0_idx).op.type()=", graph->node(w0_idx).op.type());
  BLAZE_CONDITION_THROW(graph->node(w1_idx).op.type() == "ConstantFill",
                        "graph->node(w1_idx).op.type()=", graph->node(w1_idx).op.type());

  if (node0->op.input_size() > 2) {
    int bias0_idx = node0->GetParentIdx(node0->op.input(2));
    BLAZE_CONDITION_THROW(bias0_idx >= 0, "bias0_idx=", bias0_idx);
    BLAZE_CONDITION_THROW(graph->node(bias0_idx).op.type() == "ConstantFill",
                          "graph->node(bias0_idx).op.type()=",
                          graph->node(bias0_idx).op.type());
  }
  if (node1->op.input_size() > 2) {
    int bias1_idx = node1->GetParentIdx(node1->op.input(2));
    BLAZE_CONDITION_THROW(bias1_idx >= 0, "bias1_idx=", bias1_idx);
    BLAZE_CONDITION_THROW(graph->node(bias1_idx).op.type() == "ConstantFill",
                          "graph->node(bias1_idx).op.type()=",
                          graph->node(bias1_idx).op.type());
  }
  return true;
}

void GemmGemmFusionPatternImpl::GraphRewrite(const std::vector<ArgumentHelper*>& args,
                                             std::vector<Node*>& nodes,
                                             Graph* graph) {
  Node *node0 = nodes[0], *node1 = nodes[1];

  int w0_idx = node0->GetParentIdx(node0->op.input(1));
  int bias0_idx = node0->op.input_size() <= 2 ? -1 : node0->GetParentIdx(node0->op.input(2));
  bool transposeB0 = args[0]->GetSingleArgument<bool>("transB", false);
  float alpha0 = args[0]->GetSingleArgument<float>("alpha", 1.0);
  float beta0 = args[0]->GetSingleArgument<float>("beta", 1.0);

  int w1_idx = node1->GetParentIdx(node1->op.input(1));
  int bias1_idx = node1->op.input_size() <= 2 ? -1 : node1->GetParentIdx(node1->op.input(2));
  bool transposeB1 = args[1]->GetSingleArgument<bool>("transB", false);
  float alpha1 = args[1]->GetSingleArgument<float>("alpha", 1.0);
  float beta1 = args[1]->GetSingleArgument<float>("beta", 1.0);

  //
  // alpha1 * ((alpha0 * op(X) * op(W0)) + beta0 * bias0) * op(w1) + beta1 * bias1
  // = (alpha1 * alpha0) * op(X) * op(W0) * op(W1)
  //    + beta0 * bias0 * op(W1) + beta1 * bias1
  //
  // So, the new params:
  //
  // alpha = alpha1 * alpha0
  // weight = op(W0) * op(W1)
  // bias = beta0 * bias0 * op(W1) + beta1 * bias1
  //
  *(node1->op.mutable_input(0)) = node0->op.input(0);
  
  OperatorDef new_def = node1->op;
  new_def.clear_input();
  new_def.add_input(node0->op.input(0));
  new_def.add_input(node1->op.input(1));
  new_def.clear_arg();

  Argument* new_arg = new_def.add_arg();
  new_arg->set_name("alpha");
  new_arg->set_f(alpha0 * alpha1);

  new_arg = new_def.add_arg();
  new_arg->set_name("beta");
  new_arg->set_f(1.0);

  // Calculate the new weight and bias.
  DeviceOption device_option;
  device_option.set_device_type(kCPU);
  Blob new_weight_blob(device_option);
  CalcNewWeight(&(graph->node(w0_idx).op),
                transposeB0,
                &(graph->node(w1_idx).op),
                transposeB1,
                &new_weight_blob);

  // Set ConstantFill op's value or Create a new ConstantFill node.
  if (bias0_idx >= 0 || bias1_idx >= 0) {
    // new gemm has bias
    Blob new_bias_blob(device_option);
    int bias_dtype;
    CalcNewBias(beta0,
                bias0_idx < 0 ? nullptr : &(graph->node(bias0_idx).op),
                &(graph->node(w1_idx).op),
                transposeB1,
                beta1,
                bias1_idx < 0 ? nullptr : &(graph->node(bias1_idx).op),
                &new_bias_blob,
                &bias_dtype);

    int fused_bias_idx = -1;
    if (bias1_idx >= 0) {
      fused_bias_idx = bias1_idx;
      new_def.add_input(node1->op.input(2));
    } else if (bias0_idx >= 0) {
      fused_bias_idx = bias0_idx;
      new_def.add_input(node0->op.input(2));
    } else {
      fused_bias_idx = graph->AddConstantFillNode(node1);
      new_def.add_input(node1->op.input(2));
    }
    const std::vector<TIndex> bias_shape = new_bias_blob.shape();
    ArgumentHelper::SetRepeatedArgument((graph->node(fused_bias_idx).op), "shape", bias_shape);
    ArgumentHelper::SetSingleArgument((graph->node(fused_bias_idx).op), "dtype", bias_dtype);
    TYPE_SWITCH(new_bias_blob.data_type(), DType, {
      ArgumentHelper::SetRepeatedArgument((graph->node(fused_bias_idx).op), "value",
                                          new_bias_blob.as<DType>(), new_bias_blob.size());            
    });
  }  // if (bias0_idx >= 0 || bias1_idx >= 0) {

  const std::vector<TIndex>& w_shape = new_weight_blob.shape();
  ArgumentHelper::SetRepeatedArgument(graph->node(w1_idx).op, "shape", w_shape);
  TYPE_SWITCH(new_weight_blob.data_type(), DType, {
    ArgumentHelper::SetRepeatedArgument(graph->node(w1_idx).op, "value",
                                        new_weight_blob.as<DType>(), new_weight_blob.size());
  });
  node1->op = new_def;
}

// weight = op(W0) * op(W1)
void GemmGemmFusionPatternImpl::CalcNewWeight(OperatorDef* w0,
                                              bool transpose_w0,
                                              OperatorDef* w1,
                                              bool transpose_w1,
                                              Blob* new_weight_blob) {
  ArgumentHelper w0_argument_helper(*w0);
  ArgumentHelper w1_argument_helper(*w1);

  std::vector<TIndex> w0_shape = w0_argument_helper.GetRepeatedArgument<TIndex>("shape");
  std::vector<TIndex> w1_shape = w1_argument_helper.GetRepeatedArgument<TIndex>("shape");

  BLAZE_CONDITION_THROW(w0_shape.size() == 2, "w0_shape.size()=", w0_shape.size());
  BLAZE_CONDITION_THROW(w1_shape.size() == 2, "w1_shape.size()=", w1_shape.size());

  size_t M, K0, K1, N;
  M = w0_shape[0];
  K0 = w0_shape[1];
  if (transpose_w0) {
    std::swap(M, K0);
  }
  K1 = w1_shape[0];
  N = w1_shape[1];
  if (transpose_w1) {
    std::swap(K1, N);
  }
  BLAZE_CONDITION_THROW(K0 == K1, "K0=", K0, " K1=", K1);
  size_t K = K0;

  int w0_dtype = w0_argument_helper.GetSingleArgument<int>("dtype", kFloat);
  int w1_dtype = w1_argument_helper.GetSingleArgument<int>("dtype", kFloat);
  BLAZE_CONDITION_THROW(w0_dtype == w1_dtype, "w0_dtype=", w0_dtype, " w1_dtype=", w1_dtype);

  new_weight_blob->set_data_type(DataType2PassDataType(w0_dtype));
  new_weight_blob->Reshape({ M, N});

  TYPE_SWITCH(new_weight_blob->data_type(), DType, {
    std::vector<DType> w0_data = w0_argument_helper.GetRepeatedArgument<DType>("value");
    std::vector<DType> w1_data = w1_argument_helper.GetRepeatedArgument<DType>("value");
    // calculate the fused gemm weight
    Gemm<DType, CPUContext>(transpose_w0 ? CblasTrans : CblasNoTrans,
                            transpose_w1 ? CblasTrans : CblasNoTrans,
                            M,
                            N,
                            K,
                            1.0,
                            w0_data.data(),
                            w1_data.data(),
                            0,
                            new_weight_blob->as<DType>(),
                            nullptr);
  });
}

// bias = beta0 * bias0 * op(W1) + beta1 * bias1
void GemmGemmFusionPatternImpl::CalcNewBias(float beta0,
                                            OperatorDef* bias0,
                                            OperatorDef* w1,
                                            bool transpose_w1,
                                            float beta1,
                                            OperatorDef* bias1,
                                            Blob* new_bias_blob,
                                            int* bias_dtype) {
  ArgumentHelper w1_argument_helper(*w1);
  std::vector<TIndex> w1_shape = w1_argument_helper.GetRepeatedArgument<TIndex>("shape");
  int dtype = w1_argument_helper.GetSingleArgument<int>("dtype", kFloat);
  new_bias_blob->set_data_type(DataType2PassDataType(dtype));
  *bias_dtype = dtype;

  TYPE_SWITCH(new_bias_blob->data_type(), DType, {
  std::vector<DType> w1_data = w1_argument_helper.GetRepeatedArgument<DType>("value");
  // beta0 * bias0 * op(W1) 
  if (bias0 != nullptr) {    
    ArgumentHelper bias0_argument_helper(*bias0);
    
    std::vector<TIndex> bias0_shape = bias0_argument_helper.GetRepeatedArgument<TIndex>("shape");
    int bias0_dtype = bias0_argument_helper.GetSingleArgument<int>("dtype", kFloat);
    BLAZE_CONDITION_THROW(dtype == bias0_dtype, "dtype=", dtype, " bias0_dtype=", bias0_dtype);
    std::vector<DType> bias0_data = bias0_argument_helper.GetRepeatedArgument<DType>("value");
    
    size_t M, K0, K, N;
    if (bias0_shape.size() == 1) {
      M = 1;
      K0 = bias0_shape[0];
    } else {
      M = bias0_shape[0];
      K0 = bias0_shape[1];
    }
    K = w1_shape[0];
    N = w1_shape[1];
    if (transpose_w1) {
      std::swap(K, N);
    }
    BLAZE_CONDITION_THROW(K0 == K, "K0=", K0, " K=", K);
    if (bias0_shape.size() == 1) { 
      new_bias_blob->Reshape({ N });
    } else {
      new_bias_blob->Reshape({ M, N });
    }
    // calculate beta0 * bias0 * op(W1)
    Gemm<DType, CPUContext>(CblasNoTrans,
                            transpose_w1 ? CblasTrans : CblasNoTrans,
                            M,
                            N,
                            K,
                            beta0,
                            bias0_data.data(),
                            w1_data.data(),
                            0,
                            new_bias_blob->as<DType>(),
                            nullptr);
  }
  // + (beta1 * bias1)
  if (bias1 != nullptr) {
    ArgumentHelper bias1_argument_helper(*bias1);
    
    std::vector<TIndex> bias1_shape = bias1_argument_helper.GetRepeatedArgument<TIndex>("shape");
    int bias1_dtype = bias1_argument_helper.GetSingleArgument<int>("dtype", kFloat);
    BLAZE_CONDITION_THROW(dtype == bias1_dtype, "dtype=", dtype, " bias1_dtype=", bias1_dtype);
    std::vector<DType> bias1_data = bias1_argument_helper.GetRepeatedArgument<DType>("value");

    for (size_t k = 0; k < bias1_data.size(); ++k) {
      bias1_data[k] *= beta1;
    }
    if (bias0 != nullptr) {
      BLAZE_CONDITION_THROW(UBroadcasting::DimEqual(new_bias_blob->shape(), bias1_shape),
                            "new_bias_blob and bias shape not equal");
      if (new_bias_blob->shape().size() < bias1_shape.size()) {
        // do broadcast operation
        std::vector<DType> temp(new_bias_blob->size());
        for (size_t i = 0; i < new_bias_blob->size(); ++i) {
          temp[i] = new_bias_blob->as<DType>()[i];
        }
        new_bias_blob->Reshape(bias1_shape);
        for (int k = 0; k < new_bias_blob->size() / temp.size(); ++k) {
          memcpy(new_bias_blob->as<DType>() + k * temp.size(), temp.data(), temp.size() * sizeof(DType));
        }
      }
      for (size_t k = 0; k < new_bias_blob->size(); ++k) {
        new_bias_blob->as<DType>()[k] += bias1_data[k % bias1_data.size()];
      }
    } else {
      new_bias_blob->Reshape(bias1_shape);
      memcpy(new_bias_blob->as<DType>(), bias1_data.data(), bias1_data.size() * sizeof(DType));
    }
  }
  });
}

}  // namespace blaze

