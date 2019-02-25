// Gpu Device Property 
//
#include <iostream>
#include <stdio.h>

#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

/// Print cuda device property in console.
int main(int argc, char** argv) {
#ifdef USE_CUDA
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           (int)error_id, cudaGetErrorString(error_id));
    return -1;
  }
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
    return -1;
  }
  for (int i = 0; i < deviceCount; ++i) {
    int dev = i;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("--------------------------- DEVICE %d PROPERTY ---------------------------\n", dev);
    printf(" CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf(" Total amount of global memory: (%llu bytes)\n", (unsigned long long) deviceProp.totalGlobalMem);
    printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf(" Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);
    printf(" Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf(" Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
    printf(" Warp size: %d\n", deviceProp.warpSize);
    printf(" Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf(" Maximum sizes of each dimension of a block: %d x %d x %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Maximum sizes of each dimension of a grid: %d x %d x %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" SharedMemPerMultiprocessor: %d bytes\n", deviceProp.sharedMemPerMultiprocessor);
    printf(" RegsPerMultiprocessor: %d bytes\n", deviceProp.regsPerMultiprocessor);
    printf(" MultiProcessorCount: %d\n", deviceProp.multiProcessorCount);
  }
#else
  printf("Please compile with USE_CUDA");
#endif
  return 0;
}
