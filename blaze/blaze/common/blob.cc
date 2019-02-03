/*
 * \file blob.cc 
 * \brief The blob implementation
 */
#include "blaze/common/blob.h"

#include "blaze/common/context.h"

namespace blaze {

Blob::~Blob() {
  Destroy();
}

void Blob::Destroy() {
  if (own_handle_ && this->data_) {
    const int device_id = device_option_.device_id();
    const int device_type = device_option_.device_type();
  
    blaze::Free(this->data_, capacity_ * DataTypeSize(data_type_), device_type, device_id);
    this->data_ = nullptr;
  }
}

void Copy(Blob* dst, const Blob* src, void* stream) {
  int dst_device_type = dst->device_type();
  int src_device_type = src->device_type();

  if (g_copy_function[src_device_type][dst_device_type] != nullptr) {
    g_copy_function[src_device_type][dst_device_type](
        dst->data(), src->data(), 0, 0, src->size() * DataTypeSize(src->data_type()), stream);
  }
}

}  // namespace blaze

