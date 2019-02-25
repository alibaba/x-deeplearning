/*
 * \file gru.h
 * \brief The gru operation
 */
#pragma once

#include "blaze/common/context.h"

namespace blaze {

template <typename DType, class Context>
void GRU_V1(int batch_size,
         int num_hidden,
         DType* preact,
         DType* i2h_bias,
         DType* h2h,
         DType* h2h_bias,
         int round,
         DType* y,
         Context* ctx);

}  // namespace blaze
