#ifndef CAFFE_ARRAYFIREMEM_HPP_
#define CAFFE_ARRAYFIREMEM_HPP_

#include <cstdlib>

#include "caffe/syncedmem.hpp"
#include <arrayfire.h>

namespace caffe {

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU) using ArrayFire's array
 *
 * TODO(dox): more thorough description.
 */
class ArrayFireMemory {
 public:
	ArrayFireMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(SyncedMemory::UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit ArrayFireMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(SyncedMemory::UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  ~ArrayFireMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  SyncedMemory::SyncedHead head() { return head_; }
  void set_head(SyncedMemory::SyncedHead head) { if (head_ != SyncedMemory::UNINITIALIZED) head_ = head; }
  size_t size() { return size_; }

  // Retrieves underlaying af::array implementation of memory, but in f32 format.
  // Use arrayfire cast function to cast it to different format.
  //
  // NOTE: Call to lock_arrayfire_data() has to be made after accsess through
  // arrayfire is not needed any more. Do not use raw pointer, i.e. gpu_data(), untill
  // lock_arrayfire_data() is called.
  const shared_ptr<af::array>& unlock_arrayfire_data() {
	  // Arrayfire data is by default locked since
	  // we retrieved device memory immediately after mem was created.
	  // We need to call unlock on arrayfire mem to release that lock.
	  gpu_data_->unlock();
	  return gpu_data_;
  }

  void lock_arrayfire_data() const {
	  void* new_gpu_ptr = gpu_data_->device<float>();
	  CHECK_EQ(new_gpu_ptr, gpu_ptr_);
  }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  shared_ptr<af::array> gpu_data_;
  size_t size_;
  SyncedMemory::SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;

  DISABLE_COPY_AND_ASSIGN(ArrayFireMemory);
};  // class ArrayFireMemory

}  // namespace caffe

#endif  // CAFFE_ARRAYFIREMEM_HPP_
