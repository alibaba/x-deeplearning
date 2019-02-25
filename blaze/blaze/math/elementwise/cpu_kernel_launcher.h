/*
 * \file cpu_kernel_launcher.h
 * \desc cpu kernel launcher   
 */
#pragma once

namespace blaze {

template<typename OP, class Context>
class CpuKernelLauncher {
 public:
  template<typename... Args>
  inline static void Launch(int N, const Context& context, Args... args) {
    for (int i = 0; i < N; i++) {
      OP::Map(i, args...);
    }
  }
};

} // namespace blaze
