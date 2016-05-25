#ifndef CAFFE_UTIL_MATH_FUNCTIONS_EXTRA_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_EXTRA_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

template <typename Dtype>
void caffe_cpu_copy_strided(const int N, const Dtype* X, int incx, Dtype *Y, int incy);

template <typename Dtype>
void caffe_cpu_sum(const int N, const Dtype* x, Dtype* y, int M = 0);

template <typename Dtype>
void caffe_cpu_mul_batch(const int N, const Dtype* a, const Dtype* b, Dtype* y, const int M = 0) {
	for (unsigned int batch_offset = 0; batch_offset < N; batch_offset+=M)
	 {
		for (int index = 0; index < M; ++index)
			y[index + batch_offset] = a[index + batch_offset] * b[index];

	 }
}

#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void caffe_gpu_gemm_batched(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype** A, const Dtype** B, const Dtype beta,
    Dtype** C, int batch_size);

template <typename Dtype>
void caffe_gpu_mul_batched(const int N, const Dtype* a, const Dtype* b, Dtype* y, const int M = 0);

template <typename Dtype>
void caffe_gpu_mul_split(const int N, const Dtype* a, const Dtype* b, Dtype* y, const int M, const int K, const int L);

template <typename Dtype>
void caffe_gpu_sum(const int N, const Dtype* x, Dtype* y, const int M = 0);

template <typename Dtype>
void caffe_gpu_sum(const int N, const Dtype* x, Dtype* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId = NULL);

template <typename Dtype>
void caffe_gpu_dot_batched(const int n, const Dtype* a, const Dtype* b, Dtype* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId = 0);

template <typename Dtype>
void caffe_gpu_dot_batched_mapped(const int n, const Dtype* a, const int* mapping, const Dtype* b, Dtype* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId = 0);

template <typename Dtype>
void caffe_gpu_sum_elementwise(const int N, const Dtype* x, Dtype* y, const int M);

template <typename Dtype>
void caffe_gpu_clip_lower(const int N, const Dtype lower_bound, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_clip_upper(const int N, const Dtype upper_bound, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_clip_eps(const int N, const Dtype eps_bound, const Dtype* x, Dtype* y);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_EXTRA_H_
