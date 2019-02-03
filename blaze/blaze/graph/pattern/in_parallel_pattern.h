/*
 * \file in_parallel_pattern.h 
 * \brief The in parallel pattern
 */
#include "blaze/graph/fusion_pattern.h"

#include "blaze/graph/pattern/in_parallel/parallel_gemm_fusion_pattern_impl.h"
#include "blaze/graph/pattern/in_parallel/parallel_slice_concat_fusion_pattern_impl.h"
#include "blaze/graph/pattern/in_parallel/parallel_split_reducesum_fusion_pattern_impl.h"
#include "blaze/graph/pattern/in_parallel/parallel_split_matmul_fusion_pattern_impl.h"
#include "blaze/graph/pattern/in_parallel/parallel_split_mul_fusion_pattern_impl.h"
#include "blaze/graph/pattern/in_parallel/parallel_split_softmax_fusion_pattern_impl.h"

namespace blaze {

// slice,slice,slice,...=> Concat  -> FusedSliceConcat
REGISTER_FUSION_PATTERN(FusedSliceConcat)
    .Type(kInParallel)
    .FusionOpName("FusedSliceConcat")
    .AddOpNode("slice", "Slice")
    .AddOpNode("concat", "Concat")
    .AddConnect("slice", "concat")
    .SetFusionPatternImpl(new ParallelSliceConcatFusionPatternImpl())
    .Init();

// op->Gemm/Gemm/Gemm/... -> FusedParallelGemm.
REGISTER_FUSION_PATTERN(FusedGemm)
    .Type(kInParallel)
    .FusionOpName("FusedParallelGemm")
    .AddOpNode("gemm0", "Gemm")
    .SetFusionPatternImpl(new ParallelGemmFusionPatternImpl())
    .Init();

// split->MatMul/MatMul/MatMul/... -> FusedParallelMatMul
REGISTER_FUSION_PATTERN(FusedSplitMatMul)
    .Type(kInParallel)
    .FusionOpName("FusedParallelMatMul")
    .AddOpNode("split", "Split")
    .AddOpNode("matmul", "MatMul")
    .AddConnect("split", "matmul")
    .SetFusionPatternImpl(new ParallelSplitMatMulFusionPatternImpl())
    .Init();

// split->Softmax/Softmax/Softmax/... -> Softmax/Split
REGISTER_FUSION_PATTERN(FusedSplitSoftmax)
    .Type(kInParallel)
    .AddOpNode("split", "Split")
    .AddOpNode("softmax", "Softmax")
    .AddConnect("split", "softmax")
    .SetFusionPatternImpl(new ParallelSplitSoftmaxFusionPatternImpl())
    .Init();

// split->Mul/Mul/Mul/... -> FusedParallelMul/Split
REGISTER_FUSION_PATTERN(FusedSplitMul)
    .Type(kInParallel)
    .FusionOpName("FusedParallelMul")
    .AddOpNode("split", "Split")
    .AddOpNode("mul", "Mul")
    .AddConnect("split", "mul")
    .SetFusionPatternImpl(new ParallelSplitMulFusionPatternImpl())
    .Init();

// split->ReduceSum/ReduceSum/ReduceSum/...-> FusedReduceSum/Split
REGISTER_FUSION_PATTERN(FusedReduceSum)
    .Type(kInParallel)
    .AddOpNode("split", "Split")
    .AddOpNode("reducesum", "ReduceSum")
    .AddConnect("split", "reducesum")
    .SetFusionPatternImpl(new ParallelSplitReduceSumFusionPatternImpl())
    .Init();

}  // namespace blaze
