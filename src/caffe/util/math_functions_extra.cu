#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions_extra.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/custom_cub.cuh"

namespace caffe {


template <>
void caffe_gpu_gemm_batched<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float** A, const float** B, const float beta,
    float** C, int batch_size) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemmBatched(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N, batch_size));
}

template <>
void caffe_gpu_gemm_batched<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double** A, const double** B, const double beta,
    double** C, int batch_size) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemmBatched(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N, batch_size));
}

#include <stdio.h>

// N .. size of A
// M .. size of B
// assumed N = k*M where k > 1; B is replicated k-times and Y is element-wise multiplication of A and replicated B
template <typename Dtype>
__global__ void mul_kernel_batched(const int n, const Dtype* a,
    const Dtype* b, Dtype* y, const int m) {
	unsigned int batch_offset = 0;
 {
   CUDA_KERNEL_LOOP(index, m) {
	  Dtype b_at_index = b[index];
	  for (unsigned int batch_offset = 0; batch_offset < n; batch_offset+=m)
		y[index + batch_offset] = a[index + batch_offset] * b_at_index;
  }
 }
}

template <typename Dtype>
__global__ void mul_kernel_split(const int n, const Dtype* a,
    const Dtype* b, Dtype* y, const int m, const int k, const int l) {
	unsigned int batch_offset = 0;
 {
   CUDA_KERNEL_LOOP(index, m) {

	  Dtype b_at_index = b[index];

	  int split_index = index / k;
	  int bb =  index - split_index * k;
	  int split_start_offset = l * split_index + bb;
	  int split_end_offset = l * (split_index+1) + bb;

	  for (unsigned int batch_offset = split_start_offset; batch_offset < split_end_offset; batch_offset+=k)
		y[batch_offset] = a[batch_offset] * b_at_index;
		;
  }
 }
}


template <typename Dtype>
__global__ void mul_kernel_batched_v2(const int n, const Dtype* a,
    const Dtype* b, Dtype* y, const int m) {
	unsigned int batch_offset = 0;

 {
  CUDA_KERNEL_LOOP(index, m) {
	  for (unsigned int batch_offset = 0; batch_offset < n; batch_offset+=m)
		y[index + batch_offset] = a[index + batch_offset] * b[index];
  }
 }
}


template <typename Dtype>
__global__ void mul_kernel_batched_v3(const int n, const Dtype* a,
    const Dtype* b, Dtype* y, const int m) {
	unsigned int batch_offset = 0;

 {
		const int m4 = m/4;
		const int n4 = n/4;
  CUDA_KERNEL_LOOP(index, m4) {
	  const float4 b_f4 = reinterpret_cast<const float4*>(b)[index];
	  for (unsigned int batch_offset = 0; batch_offset < n4; batch_offset+=m4) {
		  const float4 a_f4 = reinterpret_cast<const float4*>(a)[index + batch_offset];
	  	  float4 c_f4;
	  	  c_f4.x = a_f4.x * b_f4.x;
	  	  c_f4.y = a_f4.y * b_f4.y;
	  	  c_f4.z = a_f4.z * b_f4.z;
	  	  c_f4.w = a_f4.w * b_f4.w;

		  reinterpret_cast<float4*>(y)[index + batch_offset] = c_f4;
	  }
  }
 }
}

template <typename Dtype>
__global__ void mul_kernel_tmp(const int n, const Dtype a, const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    2.3*b[index];
  }
}

inline int _CAFFE_GET_BLOCKS(const int N) {
  return (N + 1024 - 1) / 1024;
}

template <>
void caffe_gpu_mul_batched<float>(const int N, const float* a,
    const float* b, float* y, const int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (M <= 0 || M > N)
	  caffe_gpu_mul<float>(N, a, b, y);
	  //mul_kernel_tmp<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, (float)0.21, b, y);
  else
	  mul_kernel_batched<float><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y, M);
}

template <>
void caffe_gpu_mul_batched<double>(const int N, const double* a,
    const double* b, double* y, const int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (M <= 0 || M > N)
	  caffe_gpu_mul<double>(N, a, b, y);
  else
	  mul_kernel_batched<double><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y, M);
}


template <>
void caffe_gpu_mul_split<float>(const int N, const float* a,
    const float* b, float* y, const int M, const int K, const int L) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel_split<float><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y, M, K, L);
}

template <>
void caffe_gpu_mul_split<double>(const int N, const double* a,
  const double* b, double* y, const int M, const int K, const int L) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel_split<double><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y, M, K, L);
}


__global__ void clip_lower_kernel_double(const int n, const double lower_bound, const double* x, double* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fmax(x[index], lower_bound);
  }
}

__global__ void clip_lower_kernel_float(const int n, const float lower_bound, const float* x, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fmaxf(x[index], lower_bound);
  }
}

template <>
void caffe_gpu_clip_lower<float>(const int N, const float lower_bound, const float* x, float* y) {
	clip_lower_kernel_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, lower_bound, x, y);
}

template <>
void caffe_gpu_clip_lower<double>(const int N, const double lower_bound, const double* x, double* y) {
	clip_lower_kernel_double<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, lower_bound, x, y);
}


__global__ void clip_upper_kernel_double(const int n, const double lower_bound, const double* x, double* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fmin(x[index], lower_bound);
  }
}

__global__ void clip_upper_kernel_float(const int n, const float lower_bound, const float* x, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = fminf(x[index], lower_bound);
  }
}

template <>
void caffe_gpu_clip_upper<float>(const int N, const float lower_bound, const float* x, float* y) {
	clip_upper_kernel_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, lower_bound, x, y);
}

template <>
void caffe_gpu_clip_upper<double>(const int N, const double lower_bound, const double* x, double* y) {
	clip_upper_kernel_double<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, lower_bound, x, y);
}



template <typename Dtype>
void caffe_gpu_clip_upper(const int N, const Dtype upper_bound, const Dtype* x, Dtype* y) {

}


template <typename Dtype>
__global__ void clip_eps_kernel(const int n, const Dtype eps_bound, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	Dtype val = x[index];
    y[index] = abs(val) > eps_bound ? val : 0;
  }
}


template <>
void caffe_gpu_clip_eps<float>(const int N, const float eps_bound, const float* x, float* y) {
	clip_eps_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, eps_bound, x, y);
}
template <>
void caffe_gpu_clip_eps<double>(const int N, const double eps_bound, const double* x, double* y) {
	clip_eps_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, eps_bound, x, y);
}


template <typename Dtype>
void caffe_gpu_sum(const int n, const Dtype* x, Dtype* y, const int m) {
	CHECK_EQ(n % m, 0);
	int num_segments = n/m;

	int* offsets = new int[num_segments + 1];

	offsets[0] = 0;

	for (int i = 0; i < num_segments; i++) offsets[i+1] = m*(i+1);

	int* offsets_d;
	CUDA_CHECK(cudaMalloc(&offsets_d, sizeof(int)*(num_segments+1)));

	caffe_gpu_memcpy(sizeof(int)*(num_segments + 1), offsets, offsets_d);

	caffe_gpu_sum(n, x, y, num_segments, offsets_d , (cudaStream_t)NULL);

	delete offsets;
}



template <typename Dtype>
void caffe_gpu_sum(const int n, const Dtype* x, Dtype* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId) {

	// DeviceSegmentedReduce in version 1.5.1 always returns temp_storage_bytes=1 and never actually uses allocated storage
	// so we can just use non-zero value for temp storage and avoid getting temp_storage_bytes size
	size_t temp_storage_bytes = 0;
	void* temp_storage_d = (void*)1;

	//CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(NULL, temp_storage_bytes, x, y,  num_segments, offsets_d, offsets_d + 1, streamId));
	//CUDA_CHECK(cudaMalloc(&temp_storage_d, temp_storage_bytes));

//	CUDA_CHECK(cub::DeviceReduce::Sum(NULL, temp_storage_bytes, x, y,  1024, streamId));
//	CUDA_CHECK(cudaMalloc(&temp_storage_d, temp_storage_bytes));

//	CUDA_CHECK(cub::DeviceReduce::Sum(temp_storage_d, temp_storage_bytes, x, y,  1024, streamId, false));

	//CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(temp_storage_d, temp_storage_bytes, x, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));
	CUDA_CHECK(custom_cub::segmentedSumWithAdd(temp_storage_d, temp_storage_bytes, x, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));

//	CUDA_CHECK(cudaFree(temp_storage_d));
}


template void caffe_gpu_sum<float>(const int n, const float* x, float* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId);
template void caffe_gpu_sum<double>(const int n, const double* x, double* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId);

template void caffe_gpu_sum<float>(const int n, const float* x, float* y, const int m);
template void caffe_gpu_sum<double>(const int n, const double* x, double* y, const int m);



template <typename Dtype>
void caffe_gpu_dot_batched(const int n, const Dtype* a, const Dtype* b, Dtype* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId) {

	// DeviceSegmentedReduce in version 1.5.1 always returns temp_storage_bytes=1 and never actually uses allocated storage
	// so we can just use non-zero value for temp storage and avoid getting temp_storage_bytes size
	size_t temp_storage_bytes = 0;
	void* temp_storage_d = (void*)1;

	custom_cub::BinaryTransformInputIterator<Dtype, custom_cub::Mul, const Dtype*, const Dtype*> input_data(a,b,custom_cub::Mul());
	CUDA_CHECK(custom_cub::segmentedSumWithAdd(temp_storage_d, temp_storage_bytes, input_data, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));
}

template void caffe_gpu_dot_batched<float>(const int n, const float* a, const float* b, float* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId);
template void caffe_gpu_dot_batched<double>(const int n, const double* a, const double* b, double* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId);



template <typename Dtype>
void caffe_gpu_dot_batched_mapped(const int n, const Dtype* a, const int* mapping, const Dtype* b, Dtype* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId) {

	// DeviceSegmentedReduce in version 1.5.1 always returns temp_storage_bytes=1 and never actually uses allocated storage
	// so we can just use non-zero value for temp storage and avoid getting temp_storage_bytes size
	size_t temp_storage_bytes = 0;
	void* temp_storage_d = (void*)1;

	// remap A based on mapping array indeces
	custom_cub::InputMappingIterator<Dtype, const Dtype*, const int*> a_mapped(a, mapping);
	// combine A and B with multiplicator (combined with sum this will effectevly perform dot product)
	custom_cub::BinaryTransformInputIterator<Dtype, custom_cub::Mul, custom_cub::InputMappingIterator<Dtype, const Dtype*, const int* >, const Dtype*> input_data(a_mapped,b,custom_cub::Mul());
	CUDA_CHECK(custom_cub::segmentedSumWithAdd(temp_storage_d, temp_storage_bytes, input_data, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));

}

template void caffe_gpu_dot_batched_mapped<float>(const int n, const float* a, const int* mapping, const float* b, float* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId);
template void caffe_gpu_dot_batched_mapped<double>(const int n, const double* a, const int* mapping, const double* b, double* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId);



// TODO: sum_elementwise should be implemented more efficently!!
template <typename Dtype>
__global__ void sum_elementwise_kernel(const int n, const Dtype* x, Dtype* y, const int m) {
  CUDA_KERNEL_LOOP(index, m) {
	  Dtype sum = 0;
	  for (unsigned int batch_offset = 0; batch_offset < n; batch_offset+=m) {
		  sum += x[index + batch_offset];
  	  }
	  y[index] = sum;
  }
}

template <>
void caffe_gpu_sum_elementwise<float>(const int N, const float* x, float* y, const int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	sum_elementwise_kernel<float><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(N, x, y, M);
}

template <>
void caffe_gpu_sum_elementwise<double>(const int N, const double* x, double* y, const int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	sum_elementwise_kernel<double><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(N, x, y, M);
}


}  // namespace caffe
