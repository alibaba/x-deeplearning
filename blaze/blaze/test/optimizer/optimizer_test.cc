/*
 * \file optimizer_test.cc
 * \brief The optimizer test unit
 */
#include "gtest/gtest.h"

#include <math.h>

#include "blaze/optimizer/optimizer.h"

namespace blaze {

TEST(TestOptimizer, RunPass) {
  Optimizer* optimizer = Optimizer::Get();
  NetDef net_def;
  NetDef net_def2 = optimizer->RunPass(net_def);
}

}  // namespace blaze


