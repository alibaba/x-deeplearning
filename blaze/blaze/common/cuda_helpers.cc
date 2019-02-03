/*!
 * \file cuda_helpers.cc
 * \brief Some cuda helper functions.
 */
#include <map>
#include <mutex>

#include "blaze/common/common_defines.h"
#include "blaze/common/cuda_helpers.h"
#include "blaze/common/exception.h"

namespace blaze {

#ifdef USE_CUDA

std::map<int, cudaDeviceProp*> device_props;
std::mutex props_mutex;

cudaDeviceProp* GetDeviceProp(int device_id) {
  std::unique_lock<std::mutex> l(props_mutex);
  auto it = device_props.find(device_id);
  if (it == device_props.end()) {
    auto prop = new cudaDeviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(prop, device_id));
    device_props[device_id] = prop;
    it = device_props.find(device_id);
  }
  return it->second;
}

#endif // USE_CUDA

}
