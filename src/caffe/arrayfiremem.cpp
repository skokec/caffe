#include "caffe/common.hpp"
#include "caffe/arrayfiremem.hpp"
#include "caffe/util/math_functions.hpp"

#include <af/cuda.h>

namespace caffe {

ArrayFireMemory::~ArrayFireMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    //CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
#endif  // CPU_ONLY
}

inline void ArrayFireMemory::to_cpu() {
  switch (head_) {
  case SyncedMemory::UNINITIALIZED:
	LOG(INFO) << "initializing cpu data";
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = SyncedMemory::HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case SyncedMemory::HEAD_AT_GPU:
#ifndef CPU_ONLY
	LOG(INFO) << "copying to cpu";
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SyncedMemory::SYNCED;
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::HEAD_AT_CPU:
  case SyncedMemory::SYNCED:
    break;
  }
}

inline void ArrayFireMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case SyncedMemory::UNINITIALIZED:
	LOG(INFO) << "initializing gpu data";
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    //CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));

    afcu::setNativeId(gpu_device_);
    gpu_data_.reset(new af::array((int)(size_ / 4), f32));
    gpu_ptr_ = gpu_data_->device<float>();

    CHECK(gpu_ptr_);

    caffe_gpu_memset(size_, 0, gpu_ptr_);

    head_ = SyncedMemory::HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case SyncedMemory::HEAD_AT_CPU:
	LOG(INFO) << "copying to gpu";
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      //CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));

      afcu::setNativeId(gpu_device_);
      gpu_data_.reset(new af::array((int)(size_ / 4), f32));
      gpu_ptr_ = gpu_data_->device<float>();

      CHECK(gpu_ptr_);

      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SyncedMemory::SYNCED;
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* ArrayFireMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void ArrayFireMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = SyncedMemory::HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* ArrayFireMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void ArrayFireMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    gpu_data_.reset();
    //CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
  gpu_ptr_ = data;
  head_ = SyncedMemory::HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* ArrayFireMemory::mutable_cpu_data() {
  to_cpu();
  head_ = SyncedMemory::HEAD_AT_CPU;
  return cpu_ptr_;
}

void* ArrayFireMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = SyncedMemory::HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void ArrayFireMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == SyncedMemory::HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SyncedMemory::SYNCED;
}
#endif

}  // namespace caffe

