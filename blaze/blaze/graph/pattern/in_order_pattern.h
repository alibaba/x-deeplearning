/*
 * \file in_order_pattern.h
 * \brief The in order pattern
 */
#include "blaze/graph/fusion_pattern.h"

#include "blaze/math/gemm.h"
#include "blaze/operator/op/constant_fill_op.h"

#include "blaze/graph/pattern/in_order/concat_concat_fusion_pattern_impl.h"
#include "blaze/graph/pattern/in_order/slice_slice_fusion_pattern_impl.h"
#include "blaze/graph/pattern/in_order/gemm_gemm_fusion_pattern_impl.h"
#include "blaze/graph/pattern/in_order/fused_parallel_mul_reduce_fusion_pattern_impl.h"

namespace blaze {

// Slice,Slice -> Slice
REGISTER_FUSION_PATTERN(SliceSlice)
    .Type(kInOrder)
    .AddOpNode("slice0", "Slice")
    .AddOpNode("slice1", "Slice")
    .AddConnect("slice0", "slice1")
    .FusionOpName("Slice")
    .SetFusionPatternImpl(new SliceSliceFusionPatternImpl())
    .Init();

// Concat,Conat->Concat
REGISTER_FUSION_PATTERN(ConcatConcat)
    .Type(kInOrder)
    .AddOpNode("concat0", "Concat")
    .AddOpNode("concat1", "Concat")
    .AddConnect("concat0", "concat1")
    .FusionOpName("Concat")
    .SetFusionPatternImpl(new ConcatConcatFusionPatternImpl())
    .Init();

// Gemm,Gemm -> Gemm
REGISTER_FUSION_PATTERN(GemmGemm)
    .Type(kInOrder)
    .AddOpNode("gemm0", "Gemm")
    .AddOpNode("gemm1", "Gemm")
    .AddConnect("gemm0", "gemm1")
    .Option("disable_reset_input", true)
    .SetFusionPatternImpl(new GemmGemmFusionPatternImpl())
    .FusionOpName("Gemm")
    .Init();

// FusedParallelMul, ReduceSum -> FusedParallelMulReduce
REGISTER_FUSION_PATTERN(FusedParallelMulReduceSum)
    .Type(kInOrder)
    .AddOpNode("fused_parallel_mul", "FusedParallelMul")
    .AddOpNode("reducesum", "ReduceSum")
    .AddConnect("fused_parallel_mul", "reducesum")
    .SetFusionPatternImpl(new FusedParallelMulReduceFusionPatternImpl())
    .FusionOpName("FusedParallelMulReduceSum")
    .Init();

}  // namespace blaze

