/*
 * \file gru_op.cu
 * \brief The gru operation
 */
#include "blaze/common/cuda_helpers.h"
#include "blaze/math/gemm.h"
#include "blaze/operator/op/gru_op.h"
#include "blaze/math/rnn/gru.h"

namespace blaze {

__global__ void GRUPadZeros(float* x, float* y, int* padded_iterations,
                            const int round, const int elts,
                            const int hidden_num) {
  const int batch_idx = blockIdx.x;
  x += batch_idx * round * elts;
  y += batch_idx * round * hidden_num;
  padded_iterations += batch_idx;
  int i;
  for (i = 0; i < round; i++) {
    auto p = x + i * elts;
    bool all_zero = true;
    for (int k = 0; k < elts; k++) {
      if (p[k] != 0) {
        all_zero = false;
        break;
      }
    }
    if (!all_zero) { break; }
    p = y + i * hidden_num;
    for (int k = 0; k < hidden_num; k++) { p[k] = 0; }
  }
  *padded_iterations = i;
}

__global__ void GRUPadZeros(float16* x, float16* y, int* padded_iterations,
                            const int round, const int elts,
                            const int hidden_num) {
  const int batch_idx = blockIdx.x;
  const int elts1 = alignN(elts, 2) / 2;
  auto no_padding = elts == elts1 * 2;
  const int hidden_num1 = alignN(hidden_num, 2) / 2;
  auto x_ = reinterpret_cast<__half2*>(x) + batch_idx * round * elts1;
  auto y_ = reinterpret_cast<__half2*>(y) + batch_idx * round * hidden_num1;
  padded_iterations += batch_idx;
  auto zero = __floats2half2_rn(0, 0);
  int i;
  for (i = 0; i < round; i++) {
    auto p = &x_[i * elts1];
    bool all_zero = true;
    for (int k = 0; k < elts1 - 1; k++) {
      if (__hbne2(p[i], zero)) {
        all_zero = false;
        break;
      }
    }
    if (all_zero) {
      all_zero = (no_padding
                  ? __hbeq2(p[elts1 - 1], zero)
                  : __low2float(p[elts1 - 1]) == 0);
    }
    if (!all_zero) { break; }
    p = &y_[i * hidden_num1];
    for (int k = 0; k < hidden_num1; k++) { p[k] = zero; }
  }
  *padded_iterations = i;
}

__global__ void GRUPrepare(unsigned int* finished, const int round) {
  for (int i = 0; i < round; i++) { finished[i] = 0; }
}

__device__ void check_readiness(bool* ready, int iter,
                                unsigned int* counter, unsigned int count) {
  if (threadIdx.x == 0) {
    *ready = iter == 0 || atomicAdd(counter, 0) == count;
  }
  __syncthreads();
}

__device__ void finish(unsigned int* counter) {
  __syncthreads();
  __threadfence();
  if (threadIdx.x == 0) { atomicAdd(counter, 1); }
}

enum GRUType {
  H2H_R, H2H_Z, H2H_H, I2H_R, I2H_Z, I2H_H
};

__forceinline__ __device__ float sigmoidf(float x) {
  return 1.0 / (1.0 + expf(-x));
}

template <const int weights_per_thread>
__device__ void load_weights(float* weights, const float* all_weights,
                             const GRUType type, const int offset,
                             const int offset_idx, const int elts,
                             const int hidden_num) {
  auto k = (type % 3 * hidden_num
            + offset
            + offset_idx * weights_per_thread * 3 * hidden_num);
  const auto max = elts * hidden_num * 3;
  #pragma unroll
  for (int i = 0; i < weights_per_thread; i++) {
    if (k < max) {
      weights[i] = all_weights[k];
      k += hidden_num * 3;
    }
  }
}

template <const int weights_per_thread>
__device__ void multiply(const float* weights, const float* inp,
                         float* out, const GRUType type, const int iter,
                         const int offset_idx, const int elts) {
  float res = 0;
  auto k = weights_per_thread * offset_idx;
  #pragma unroll
  for (int i = 0; i < weights_per_thread; i++) {
    if (k < elts) {
      res += weights[i] * (type < 3 && iter == 0 ? 0 : inp[k]);
      k++;
    }
  }
  out[threadIdx.x] = res;
}

__device__ void sum(const float* vals, float* out,
                    const int offset_idx, const int threads_per_slot) {
  if (offset_idx != 0) { return; }
  float res = 0;
  auto k = threadIdx.x;
  for (int i = 0; i < threads_per_slot; i++) { res += vals[k++]; }
  out[threadIdx.x] = res;
}

__device__ void calc_final(const float* vals, float* out, GRUType type,
                           const int offset, const int offset_idx,
                           const int threads_per_slot, const float prev_h,
                           const float hbr, const float hbz, const float hbh,
                           const float ibr, const float ibz, const float ibh) {
  if (type != 0 || offset_idx != 0) { return; }
  auto k = threadIdx.x;
  auto r1 = vals[k];
  k += threads_per_slot;
  auto z1 = vals[k];
  k += threads_per_slot;
  auto h1 = vals[k];
  k += threads_per_slot;
  auto r0 = vals[k];
  k += threads_per_slot;
  auto z0 = vals[k];
  k += threads_per_slot;
  auto h0 = vals[k];
  auto r2 = sigmoidf(r0 + ibr + r1 + hbr);
  auto z2 = sigmoidf(z0 + ibz + z1 + hbz);
  auto h2 = tanh(h0 + ibh + r2 * (h1 + hbh));
  out[offset] = (1 - z2) * h2 + z2 * prev_h;
}

template <const int weights_per_thread>
__global__ void GRUKernel(const float* x, const float* h2h,
                          const float* h2h_bias, const float* i2h,
                          const float* i2h_bias, float* y,
                          unsigned int* finished, const int batch_size,
                          const int round, const int elts,
                          const int hidden_num, const int* padded_iterations) {
  const int total_per_slot_0 = alignN(elts, gru_weights_per_thread);
  const int threads_per_slot_0 = total_per_slot_0 / weights_per_thread;
  const int total_per_slot = total_per_slot_0 * 6;
  const int threads_per_slot = total_per_slot / weights_per_thread;
  const int slot_per_block = (alignN(threads_per_slot, gru_threads_per_block)
                              / threads_per_slot);
  const int threads_per_block = slot_per_block * threads_per_slot;
  if (threadIdx.x >= threads_per_block) { return; }
  const int global_slot_idx = (blockIdx.x * slot_per_block
                               + threadIdx.x / threads_per_slot);
  const int total_slots = elts * 6 * hidden_num * batch_size;
  if (global_slot_idx >= total_slots) { return; }
  const int batch_idx = global_slot_idx / hidden_num;
  const int offset = global_slot_idx % hidden_num;
  const int offset_idx = threadIdx.x % threads_per_slot_0;
  const GRUType type = static_cast<GRUType>(
    threadIdx.x % threads_per_slot / threads_per_slot_0);
  x += batch_idx * round * elts;
  y += batch_idx * round * hidden_num;
  float weights[weights_per_thread];
  auto all_weights = type < 3 ? h2h : i2h;
  load_weights<weights_per_thread>(
    weights, all_weights, type, offset, offset_idx, elts, hidden_num);
  const float hbr = h2h_bias[offset];
  const float hbz = h2h_bias[offset + hidden_num];
  const float hbh = h2h_bias[offset + hidden_num * 2];
  const float ibr = i2h_bias[offset];
  const float ibz = i2h_bias[offset + hidden_num];
  const float ibh = i2h_bias[offset + hidden_num * 2];
  const int padded_iteration = padded_iterations[batch_idx];
  extern __shared__ float vals[];
  __shared__ bool ready[1];
  for (int iter = 0; iter < round; ) {
    check_readiness(ready, iter, &finished[iter - 1], gridDim.x);
    if (!ready[0]) { continue; }
    if (iter >= padded_iteration) {
      auto inp = (type < 3
                  ? y + (iter - 1) * hidden_num
                  : x + iter * elts);
      multiply<weights_per_thread>(
        weights, inp, vals, type, iter, offset_idx, elts);
      __syncthreads();
      sum(vals, vals, offset_idx, threads_per_slot_0);
      __syncthreads();
      float prev_h;
      if (offset_idx == 0 && type == 0) {
        prev_h = (iter == 0 ? 0 : y[(iter - 1) * hidden_num + offset]);
      }
      calc_final(vals, &y[iter * hidden_num], type, offset, offset_idx,
                 threads_per_slot_0, prev_h, hbr, hbz, hbh, ibr, ibz, ibh);
    }
    finish(&finished[iter]);
    iter++;
  }
}

#if CUDA_VERSION >= 9000

__forceinline__ __device__ __half __sigmoidh(__half x) {
  return __hdiv(__float2half(1),
                __hadd(__float2half(1), hexp(__hneg(x))));
}

__forceinline__ __device__ __half __tanhh(__half x) {
  __half v = hexp(__hmul(__float2half(2), x));
  return __hdiv(__hsub(v, __float2half(1)),
                __hadd(v, __float2half(1)));
}

#else // CUDA_VERSION >= 9000

__forceinline__ __device__ __half __sigmoidh(__half x) {
  return hdiv(__float2half(1),
              __hadd(__float2half(1), hexp(__hneg(x))));
}

__forceinline__ __device__ __half __tanhh(__half x) {
  __half v = hexp(__hmul(__float2half(2), x));
  return hdiv(__hsub(v, __float2half(1)),
              __hadd(v, __float2half(1)));
}

#endif // CUDA_VERSION >= 9000

template <const int weights_per_thread>
__device__ void load_weights(__half2* weights, const __half* all_weights,
                             const GRUType type, const int offset,
                             const int offset_idx, const int elts,
                             const int hidden_num) {
  const auto base = alignN(hidden_num, 2);
  auto k = (type % 3 * base
            + offset
            + offset_idx * weights_per_thread * 3 * base);
  const auto max = elts * hidden_num * 3;
  bool partial;
  __half weight[2];
  int i;
  #pragma unroll
  for (i = 0; i < weights_per_thread; i++) {
    if (k >= max) { break; }
    weight[i % 2] = all_weights[k];
    k += hidden_num * 3;
    if (i % 2) { weights[i / 2] = *reinterpret_cast<__half2*>(weight); }
    partial = !(i % 2);
  }
  if (partial) { weights[i / 2] = *reinterpret_cast<__half2*>(weight); }
}

template <const int weights_per_thread>
__device__ void multiply(const __half2* weights, const __half* inp,
                         __half* out, const GRUType type, const int iter,
                         const int offset_idx, const int elts) {
  __half2 res = __floats2half2_rn(0, 0);
  auto k = weights_per_thread * offset_idx;
  auto p = reinterpret_cast<const __half2*>(&inp[k]);
  #pragma unroll
  for (int i = 0; i < weights_per_thread / 2; i++) {
    if (type < 3 && iter == 0) { continue; }
    if (k < elts - 1) {
      __half2 v = *(p++);
      res = __hfma2(weights[i], v, res);
      k += 2;
    } else if (k < elts) {
      __half v0 = *reinterpret_cast<const __half*>(p);
      __half2 v = __halves2half2(v0, __float2half(0));
      res = __hfma2(weights[i], v, res);
      break;
    }
  }
  out[threadIdx.x] = __hadd(__low2half(res), __high2half(res));
}

__device__ void sum(const __half* vals, __half* out,
                    const int offset_idx, const int threads_per_slot) {
  if (offset_idx != 0) { return; }
  __half2 res = __floats2half2_rn(0, 0);
  auto p = reinterpret_cast<const __half2*>(&vals[threadIdx.x]);
  for (int i = 0; i < threads_per_slot; i += 2) {
    if (i < threads_per_slot - 1) {
      __half2 v = *(p++);
      res = __hadd2(res, v);
    } else {
      __half v0 = *reinterpret_cast<const __half*>(p);
      __half2 v = __halves2half2(v0, __float2half(0));
      res = __hadd2(res, v);
    }
  }
  out[threadIdx.x] = __hadd(__low2half(res), __high2half(res));
}

__device__ void calc_final(const __half* vals, __half* out, GRUType type,
                           const int offset, const int offset_idx,
                           const int threads_per_slot, const __half prev_h,
                           const __half hbr, const __half hbz,
                           const __half hbh, const __half ibr,
                           const __half ibz, const __half ibh) {
  if (type != 0 || offset_idx != 0) { return; }
  auto k = threadIdx.x;
  auto r1 = vals[k];
  k += threads_per_slot;
  auto z1 = vals[k];
  k += threads_per_slot;
  auto h1 = vals[k];
  k += threads_per_slot;
  auto r0 = vals[k];
  k += threads_per_slot;
  auto z0 = vals[k];
  k += threads_per_slot;
  auto h0 = vals[k];
  auto r2 = __sigmoidh(__hadd(r0, __hadd(ibr, __hadd(r1, hbr))));
  auto z2 = __sigmoidh(__hadd(z0, __hadd(ibz, __hadd(z1, hbz))));
  auto h2 = __tanhh(__hadd(h0,
                           __hadd(ibh,
                                  __hmul(r2,
                                         __hadd(h1, hbh)))));
  out[offset] = __hadd(__hmul(__hsub(__float2half(1), z2), h2),
                       __hmul(z2, prev_h));
}

template <const int weights_per_thread>
__global__ void GRUKernel(const float16* x, const float16* h2h,
                          const float16* h2h_bias, const float16* i2h,
                          const float16* i2h_bias, float16* y,
                          unsigned int* finished, const int batch_size,
                          const int round, const int elts,
                          const int hidden_num, const int* padded_iterations) {
  const int total_per_slot_0 = alignN(elts, gru_weights_per_thread);
  const int threads_per_slot_0 = total_per_slot_0 / weights_per_thread;
  const int total_per_slot = total_per_slot_0 * 6;
  const int threads_per_slot = total_per_slot / weights_per_thread;
  const int slot_per_block = (alignN(threads_per_slot, gru_threads_per_block)
                              / threads_per_slot);
  const int threads_per_block = slot_per_block * threads_per_slot;
  if (threadIdx.x >= threads_per_block) { return; }
  const int global_slot_idx = (blockIdx.x * slot_per_block
                               + threadIdx.x / threads_per_slot);
  const int total_slots = hidden_num * batch_size;
  if (global_slot_idx >= total_slots) { return; }
  const int batch_idx = global_slot_idx / hidden_num;
  const int offset = global_slot_idx % hidden_num;
  const int offset_idx = threadIdx.x % threads_per_slot_0;
  const GRUType type = static_cast<GRUType>(
    threadIdx.x % threads_per_slot / threads_per_slot_0);
  const __half* x_ = reinterpret_cast<const __half*>(
    x + batch_idx * round * alignN(elts, 2));
  __half* y_ = reinterpret_cast<__half*>(
    y + batch_idx * round * alignN(hidden_num, 2));
  const __half* h2h_ = reinterpret_cast<const __half*>(h2h);
  const __half* i2h_ = reinterpret_cast<const __half*>(i2h);
  const __half* h2h_bias_ = reinterpret_cast<const __half*>(h2h_bias);
  const __half* i2h_bias_ = reinterpret_cast<const __half*>(i2h_bias);
  __half2 weights[weights_per_thread / 2];
  auto all_weights = type < 3 ? h2h_ : i2h_;
  load_weights<weights_per_thread>(
    weights, all_weights, type, offset, offset_idx, elts, hidden_num);
  const int hidden_num1 = alignN(hidden_num, 2);
  const __half hbr = h2h_bias_[offset];
  const __half hbz = h2h_bias_[offset + hidden_num1];
  const __half hbh = h2h_bias_[offset + hidden_num1 * 2];
  const __half ibr = i2h_bias_[offset];
  const __half ibz = i2h_bias_[offset + hidden_num1];
  const __half ibh = i2h_bias_[offset + hidden_num1 * 2];
  const int padded_iteration = padded_iterations[batch_idx];
  extern __shared__ __half half_vals[];
  __shared__ bool ready[1];
  for (int iter = 0; iter < round; ) {
    check_readiness(ready, iter, &finished[iter - 1], gridDim.x);
    if (!ready[0]) { continue; }
    if (iter >= padded_iteration) {
      auto inp = (type < 3
                  ? y_ + (iter - 1) * alignN(hidden_num, 2)
                  : x_ + iter * alignN(elts, 2));
      multiply<weights_per_thread>(
        weights, inp, half_vals, type, iter, offset_idx, elts);
      __syncthreads();
      sum(half_vals, half_vals, offset_idx, threads_per_slot_0);
      __syncthreads();
      __half prev_h;
      if (offset_idx == 0 && type == 0) {
        prev_h = (iter == 0
                  ? __float2half(0)
                  : y_[(iter - 1) * alignN(hidden_num, 2) + offset]);
      }
      calc_final(half_vals, &y_[iter * alignN(hidden_num, 2)],
                 type, offset, offset_idx, threads_per_slot_0, prev_h,
                 hbr, hbz, hbh, ibr, ibz, ibh);
    }
    finish(&finished[iter]);
    iter++;
  }
}

template <>
bool GRUOp<CUDAContext>::RunOnDevice() {
  Blob* x = this->Input(0);
  TYPE_SWITCH_ON_CUDA(x->data_type(), DType, {
    GRUParam<DType> params;
    Setup(&params);
    const int hidden_num = params.elts;
    const int total_per_slot_0 = alignN(params.elts, gru_weights_per_thread);
    // 3 for input and 3 for history, so 6 in total
    const int total_per_slot = total_per_slot_0 * 6;
    const int threads_per_slot = total_per_slot / gru_weights_per_thread;
    const int slot_per_block = (alignN(threads_per_slot, gru_threads_per_block)
                                / threads_per_slot);
    const int block_count = (alignN(hidden_num, slot_per_block)
                             / slot_per_block * params.batch_size);
    int* padded_iterations = reinterpret_cast<int*>(params.act);
    const size_t cache_size =
      sizeof(DType) * slot_per_block * threads_per_slot;
    GRUPadZeros<<<params.batch_size, 1, 0, context_.cuda_stream()>>>(
      params.x, params.y, padded_iterations, params.round,
      params.elts, hidden_num);
    CUDA_CHECK(cudaGetLastError());
    GRUPrepare<<<1, 1, 0, context_.cuda_stream()>>>(
      params.finished, params.round);
    CUDA_CHECK(cudaGetLastError());
    GRUKernel<gru_weights_per_thread><<<
      block_count, gru_threads_per_block, cache_size, context_.cuda_stream()>>>(
        params.x, params.h2h, params.h2h_bias, params.i2h, params.i2h_bias,
        params.y, params.finished, params.batch_size, params.round,
        params.elts, params.elts, padded_iterations);
    CUDA_CHECK(cudaGetLastError());
  });
  return true;
}

REGISTER_CUDA_OPERATOR(GRU, GRUOp<CUDAContext>);

}  // namespace blaze

