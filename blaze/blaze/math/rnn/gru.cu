/*
 * \file gru.cu
 * \brief The gru operation
 */
#include "blaze/common/cuda_helpers.h"
#include "blaze/math/gemm.h"

#include "blaze/math/rnn/gru.h"

namespace blaze {

template <typename DType>
__forceinline__ __device__ DType sigmoidf(DType x) {
  return 1.0 / (1.0 + expf(-x));
}

template<typename DType>
__global__ void GRUKernel_V1(const DType* i2hpreact,
        const DType* i2hBias,
        const DType* h2hWeight,
        const DType* h2hBias,
        DType* output,
        int round) {
    __shared__ bool gru_sdata[1];
    if (threadIdx.x == 0) {
        gru_sdata[0] = false;
    }
    int taskId = blockIdx.x;
    int dim = blockDim.x;
    int dimIdx = threadIdx.x;
    int dimIdxR = threadIdx.x;
    int dimIdxZ = threadIdx.x + dim;
    int dimIdxH = threadIdx.x + dim * 2;
    i2hpreact = i2hpreact + taskId * dim * 3 * round;
    DType hsteppreactR;
    DType hsteppreactZ;
    DType hsteppreactH;
    output = output + taskId * dim * round;
    DType lastH = 0.0f;
    int h2hWeightOffset = dim * 3;
    bool checkDone = false;
    for (int i = 0; i < round; i++) {
        if (!checkDone) {
            if (i2hpreact[dimIdxR] != 0) {
                gru_sdata[0] = true;
            }
            __syncthreads();
            checkDone = gru_sdata[0];
        }
        if (checkDone) {
            hsteppreactR = h2hBias[dimIdxR];
            hsteppreactZ = h2hBias[dimIdxZ];
            hsteppreactH = h2hBias[dimIdxH];
            if (i > 0) {
                __syncthreads();
                for (int j = 0; j < dim; j++) {
                    hsteppreactR += h2hWeight[h2hWeightOffset * j + dimIdxR] * output[j];
                    hsteppreactZ += h2hWeight[h2hWeightOffset * j + dimIdxZ] * output[j];
                    hsteppreactH += h2hWeight[h2hWeightOffset * j + dimIdxH] * output[j];
                }
                output += dim;
            }
            DType r = sigmoidf(i2hpreact[dimIdxR] + i2hBias[dimIdxR] + hsteppreactR);
            DType z = sigmoidf(i2hpreact[dimIdxZ] + i2hBias[dimIdxZ] + hsteppreactZ);
            DType h = tanh(i2hpreact[dimIdxH] + i2hBias[dimIdxH] + r * hsteppreactH);
            lastH = (1 - z) * h + z * lastH;
            output[dimIdx] = lastH;
        } else {
            if (i > 0) {
                output += dim;
            }
            output[dimIdx] = 0.0f;
        }
        i2hpreact += dim * 3;
    }
}

template <>
void GRU_V1<float, CUDAContext>(int batch_size,
        int num_hidden,
        float* preact,
        float* i2h_bias,
        float* h2h,
        float* h2h_bias,
        int round,
        float* y,
        CUDAContext* ctx) {
    GRUKernel_V1<float><<<batch_size, num_hidden, 0,
         ctx->cuda_stream()>>>(preact, i2h_bias, h2h, h2h_bias, y, round);
}

}  // namespace blaze
