/*!
 * \file pass_register.cc
 * \brief The pass register
 */
#include "blaze/optimizer/passes/fusion_pass.h"
#include "blaze/optimizer/passes/eliminate_pass.h"
#include "blaze/optimizer/passes/concat_reduce_swap_pass.h"
#include "blaze/optimizer/passes/constant_pre_compute_pass.h"
#include "blaze/optimizer/passes/xdl_sparse_fusion_pass.h"
#include "blaze/optimizer/passes/gemm_pass.h"

namespace blaze {

// ----- The following are sparse pass optimization ----
// xdl sparse fusion pass
REGISTER_PASS(XdlSparseFusionPass).Name("XdlSparseFusionPass")
    .Type(kGraph);

// ----- The following are dense pass optimization ----
// constant pre compute pass
REGISTER_PASS(ConstantPreComputePass).Name("ConstantPreCompute")
    .Type(kGraph);
// elimiating pass
REGISTER_PASS(EliminatePass).Name("EliminatePass")
    .Type(kGraph);
// concat-reduce swap pass
REGISTER_PASS(ConcatReduceSwapPass).Name("ConcatReduceSwapPass")
    .Type(kGraph);
// gemm pass
REGISTER_PASS(GemmPass).Name("GemmPass")
    .Type(kGraph);
// Fusion pass
REGISTER_PASS(FusionPass).Name("FusionPass")
    .Type(kGraph);

}  // namespace blaze

