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


void caffe_gpu_memcpy_async(const size_t N, const void* X, void* Y, cudaStream_t streamId) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpyAsync(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

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


template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set_async(const int N, const Dtype alpha, Dtype* Y, cudaStream_t streamId) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemsetAsync(Y, 0, sizeof(Dtype) * N, streamId));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(
      N, alpha, Y);
}

template void caffe_gpu_set_async<int>(const int N, const int alpha, int* Y, cudaStream_t streamId);
template void caffe_gpu_set_async<float>(const int N, const float alpha, float* Y, cudaStream_t streamId);
template void caffe_gpu_set_async<double>(const int N, const double alpha, double* Y, cudaStream_t streamId);


#include <stdio.h>

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

// N .. size of A
// M .. size of B
// assumed N = k*M where k > 1; B is replicated k-times and Y is element-wise multiplication of A and replicated B
template <typename Dtype>
__global__ void mul_kernel_batched(const int n, const Dtype* a,
    const Dtype* b, Dtype* y, const int m) {
 {
   CUDA_KERNEL_LOOP(index, m) {
	  Dtype b_at_index = b[index];
	  for (unsigned int batch_offset = 0; batch_offset < n; batch_offset+=m)
		y[index + batch_offset] = a[index + batch_offset] * b_at_index;
  }
 }
}

template <>
void caffe_gpu_mul_batched<float>(const int N, const float* a,
    const float* b, float* y, const int M, cudaStream_t streamId) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (M <= 0 || M > N)
	  //caffe_gpu_mul<float>(N, a, b, y, streamId);
  	  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, a, b, y);
  else
	  mul_kernel_batched<float><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, a, b, y, M);
}

template <>
void caffe_gpu_mul_batched<double>(const int N, const double* a,
    const double* b, double* y, const int M, cudaStream_t streamId) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (M <= 0 || M > N)
	  //caffe_gpu_mul<double>(N, a, b, y, streamId);
	  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, a, b, y);
  else
	  mul_kernel_batched<double><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, a, b, y, M);
}


template <typename Dtype>
__global__ void mul_kernel_split(const int n, const Dtype* a,
    const Dtype* b, Dtype* y, const int m, const int k, const int l) {
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

template <>
void caffe_gpu_mul_split<float>(const int N, const float* a,
    const float* b, float* y, const int M, const int K, const int L, cudaStream_t streamId) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel_split<float><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, a, b, y, M, K, L);
}

template <>
void caffe_gpu_mul_split<double>(const int N, const double* a,
  const double* b, double* y, const int M, const int K, const int L, cudaStream_t streamId) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel_split<double><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, a, b, y, M, K, L);
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
void caffe_gpu_clip_lower<float>(const int N, const float lower_bound, const float* x, float* y, cudaStream_t streamId) {
	clip_lower_kernel_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, lower_bound, x, y);
}

template <>
void caffe_gpu_clip_lower<double>(const int N, const double lower_bound, const double* x, double* y, cudaStream_t streamId) {
	clip_lower_kernel_double<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, lower_bound, x, y);
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
void caffe_gpu_clip_upper<float>(const int N, const float upper_bound, const float* x, float* y, cudaStream_t streamId) {
	clip_upper_kernel_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, upper_bound, x, y);
}

template <>
void caffe_gpu_clip_upper<double>(const int N, const double upper_bound, const double* x, double* y, cudaStream_t streamId) {
	clip_upper_kernel_double<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, upper_bound, x, y);
}


template <typename Dtype>
__global__ void clip_eps_kernel(const int n, const Dtype eps_bound, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	Dtype val = x[index];
    y[index] = abs(val) > eps_bound ? val : 0;
  }
}


template <>
void caffe_gpu_clip_eps<float>(const int N, const float eps_bound, const float* x, float* y, cudaStream_t streamId) {
	clip_eps_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, eps_bound, x, y);
}
template <>
void caffe_gpu_clip_eps<double>(const int N, const double eps_bound, const double* x, double* y, cudaStream_t streamId) {
	clip_eps_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, eps_bound, x, y);
}



__global__ void round_kernel_float(const int n, const float* x, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = llroundf(x[index]);
  }
}

__global__ void round_kernel_double(const int n, const double* x, double* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = llround(x[index]);
  }
}


template <>
void caffe_gpu_round<float>(const int N, const float* x, float* y, cudaStream_t streamId) {
	round_kernel_float<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, x, y);
}
template <>
void caffe_gpu_round<double>(const int N,const double* x, double* y, cudaStream_t streamId) {
	round_kernel_double<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, x, y);
}

template <typename Dtype>
void caffe_gpu_sum(const int n, const Dtype* x, Dtype* y, const int m, cudaStream_t streamId) {
	CHECK_EQ(n % m, 0);
	int num_segments = n/m;

	int* offsets = new int[num_segments + 1];

	offsets[0] = 0;

	for (int i = 0; i < num_segments; i++) offsets[i+1] = m*(i+1);

	int* offsets_d;
	CUDA_CHECK(cudaMalloc(&offsets_d, sizeof(int)*(num_segments+1)));

	caffe_gpu_memcpy_async(sizeof(int)*(num_segments + 1), offsets, offsets_d);

	caffe_gpu_sum(n, x, y, num_segments, offsets_d, false, streamId);

	delete offsets;
}



template <typename Dtype>
void caffe_gpu_sum(const int n, const Dtype* x, Dtype* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId) {

	// DeviceSegmentedReduce in version 1.5.1 always returns temp_storage_bytes=1 and never actually uses allocated storage
	// so we can just use non-zero value for temp storage and avoid getting temp_storage_bytes size
	size_t temp_storage_bytes = 0;
	void* temp_storage_d = (void*)1;

	if (with_add)
		CUDA_CHECK(custom_cub::segmentedSumWithAdd(temp_storage_d, temp_storage_bytes, x, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));
	else
		CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(temp_storage_d, temp_storage_bytes, x, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));
}


template void caffe_gpu_sum<float>(const int n, const float* x, float* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId);
template void caffe_gpu_sum<double>(const int n, const double* x, double* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId);

template void caffe_gpu_sum<float>(const int n, const float* x, float* y, const int m, cudaStream_t streamId);
template void caffe_gpu_sum<double>(const int n, const double* x, double* y, const int m, cudaStream_t streamId);



template <typename Dtype>
void caffe_gpu_dot_batched(const int n, const Dtype* a, const Dtype* b, Dtype* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId) {

	// DeviceSegmentedReduce in version 1.5.1 always returns temp_storage_bytes=1 and never actually uses allocated storage
	// so we can just use non-zero value for temp storage and avoid getting temp_storage_bytes size
	size_t temp_storage_bytes = 0;
	void* temp_storage_d = (void*)1;

	custom_cub::BinaryTransformInputIterator<Dtype, custom_cub::Mul, const Dtype*, const Dtype*> input_data(a,b,custom_cub::Mul());
	if (with_add)
		CUDA_CHECK(custom_cub::segmentedSumWithAdd(temp_storage_d, temp_storage_bytes, input_data, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));
	else
		CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(temp_storage_d, temp_storage_bytes, input_data, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));
}

template void caffe_gpu_dot_batched<float>(const int n, const float* a, const float* b, float* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId);
template void caffe_gpu_dot_batched<double>(const int n, const double* a, const double* b, double* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId);



template <typename Dtype>
void caffe_gpu_dot_batched_mapped(const int n, const Dtype* a, const int* mapping, const Dtype* b, Dtype* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId) {

	// DeviceSegmentedReduce in version 1.5.1 always returns temp_storage_bytes=1 and never actually uses allocated storage
	// so we can just use non-zero value for temp storage and avoid getting temp_storage_bytes size
	size_t temp_storage_bytes = 0;
	void* temp_storage_d = (void*)1;

	// remap A based on mapping array indeces
	custom_cub::InputMappingIterator<Dtype, const Dtype*, const int*> a_mapped(a, mapping);
	// combine A and B with multiplicator (combined with sum this will effectevly perform dot product)
	custom_cub::BinaryTransformInputIterator<Dtype, custom_cub::Mul, custom_cub::InputMappingIterator<Dtype, const Dtype*, const int* >, const Dtype*> input_data(a_mapped,b,custom_cub::Mul());
	if (with_add)
		CUDA_CHECK(custom_cub::segmentedSumWithAdd(temp_storage_d, temp_storage_bytes, input_data, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));
	else
		CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(temp_storage_d, temp_storage_bytes, input_data, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));

}

template void caffe_gpu_dot_batched_mapped<float>(const int n, const float* a, const int* mapping, const float* b, float* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId);
template void caffe_gpu_dot_batched_mapped<double>(const int n, const double* a, const int* mapping, const double* b, double* y, const int num_segments, int* offsets_gpu, bool with_add, cudaStream_t streamId);



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
void caffe_gpu_sum_elementwise<float>(const int N, const float* x, float* y, const int M, cudaStream_t streamId) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	sum_elementwise_kernel<float><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, x, y, M);
}

template <>
void caffe_gpu_sum_elementwise<double>(const int N, const double* x, double* y, const int M, cudaStream_t streamId) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	sum_elementwise_kernel<double><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(N, x, y, M);
}


#define OFFSET3(k,j,i, num_k, num_j, num_i) ((((k)) * (num_j) + (j))*(num_i) + (i) )
#define OFFSET4(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

template <typename Dtype>
__global__ void transposeKernel(Dtype *odata, const Dtype *idata, const int I, const int J, const int K, const int L)
{
	CUDA_KERNEL_LOOP(index, I*J*K) {
		int idx1 = index;
		int idx2 = idx1 / (I);
		int idx3 = idx2 / (J);
		int idx4 = idx2 / (K);

		int i = idx1 % I;
		int j = idx2 % J;
		int k = idx2 % K;
		int l = idx4 % 1;

		// transpose from [L,"K,J",I] to [L,"J,K",I]
		odata[OFFSET4(l, j, k, i, L, J, K, I)] = idata[OFFSET4(l, k, j, i, L, K, J, I)];
	}
}

template <typename Dtype>
void caffe_gpu_transpose(const int I, const int J, const int K, const int L, const Dtype* X, Dtype* Y, cudaStream_t streamId) {
	// transpose middle dimensions of matrix i.e. from [L x (K x J) x I] to [L x (J x K) x I]
    transposeKernel<Dtype><<<CAFFE_GET_BLOCKS(I*J*K), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(Y, X, I, J, K, L);
}

template void caffe_gpu_transpose<float>(const int I, const int J, const int K, const int L, const float* X, float* Y, cudaStream_t streamId);
template void caffe_gpu_transpose<double>(const int I, const int J, const int K, const int L, const double* X, double* Y, cudaStream_t streamId);




}  // namespace caffe
