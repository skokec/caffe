#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
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

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}



template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}


template <typename Dtype>
__global__ void add_elementwise_kernel(const int n, const Dtype* x, Dtype* y, const int m) {
  CUDA_KERNEL_LOOP(index, n) {
	  Dtype acum = 0;
	  for (int i = index; i < m + index; i++)
		  acum += x[i];
	  y[index] = acum;
  }
}


template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}


template <>
void caffe_gpu_add_elementwise<float>(const int N, const float* x, float* y, const int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	add_elementwise_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, x, y, M );
}

template <>
void caffe_gpu_add_elementwise<double>(const int N, const double* x, double* y, const int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	add_elementwise_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
	  N, x, y, M);
}


template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}
#include <stdio.h>

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
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y, const int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (M <= 0 || M > N)
	  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
  else
	  mul_kernel_batched<float><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y, M);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y, const int M) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  if (M <= 0 || M > N)
	  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
  else
	  mul_kernel_batched<double><<<CAFFE_GET_BLOCKS(M), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y, M);
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
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

__global__ void popc_kernel(const int n, const float* a,
    const float* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popc(static_cast<uint32_t>(a[index]) ^
                      static_cast<uint32_t>(b[index]));
  }
}

__global__ void popcll_kernel(const int n, const double* a,
    const double* b, uint8_t* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = __popcll(static_cast<uint64_t>(a[index]) ^
                      static_cast<uint64_t>(b[index]));
  }
}

template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
                                  const float* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popc_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
                                   const double* y) {
  // TODO: Fix caffe_gpu_hamming_distance (see failing unit test
  // TestHammingDistanceGPU in test_math_functions.cpp).
  NOT_IMPLEMENTED;
  thrust::device_vector<uint8_t> popcounts(n);
  // NOLINT_NEXT_LINE(whitespace/operators)
  popcll_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y, thrust::raw_pointer_cast(popcounts.data()));
  return thrust::reduce(popcounts.begin(), popcounts.end(),
                        /* NOLINT_NEXT_LINE(build/include_what_you_use) */
                        (uint32_t) 0, thrust::plus<uint32_t>());
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}


#include <cub/cub/cub.cuh>

using namespace cub;

/**
 * Segmented reduction that uses d_out values as intialization values (one block per segment)
 */
template <
    typename                ChainedPolicyT,             ///< Chained tuning policy
    typename                InputIteratorT,             ///< Random-access input iterator type for reading input items \iterator
    typename                OutputIteratorT,            ///< Output iterator type for recording the reduced aggregate \iterator
    typename                OffsetT,                    ///< Signed integer type for global offsets
    typename                ReductionOpT,               ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename                T>                          ///< Data element type that is convertible to the \p value type of \p InputIteratorT
__launch_bounds__ (int(ChainedPolicyT::ActivePolicy::ReducePolicy::BLOCK_THREADS))
__global__ void MyDeviceSegmentedReduceWithInitKernel(
    InputIteratorT          d_in,                       ///< [in] Pointer to the input sequence of data items
    OutputIteratorT         d_out,                      ///< [out] Pointer to the output aggregate
    int                     *d_begin_offsets,           ///< [in] %Devic-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
    int                     *d_end_offsets,             ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
    int                     num_segments,               ///< [in] The number of segments that comprise the sorting data
    ReductionOpT            reduction_op)               ///< [in] Binary reduction functor

{
    // Thread block type for reducing input tiles
    typedef AgentReduce<
            typename ChainedPolicyT::ActivePolicy::ReducePolicy,
            InputIteratorT,
            OffsetT,
            ReductionOpT>
        AgentReduceT;

    // Shared memory storage
    __shared__ typename AgentReduceT::TempStorage temp_storage;

    OffsetT segment_begin   = d_begin_offsets[blockIdx.x];
    OffsetT segment_end     = d_end_offsets[blockIdx.x];

    // Check if empty problem
    if (segment_begin == segment_end)
    {
        return;
    }

    // Consume input tiles
    T block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op).ConsumeRange(
        segment_begin,
        segment_end);

    // Normalize as needed
    NormalizeReductionOutput(block_aggregate, segment_begin, d_in);

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = reduction_op(d_out[blockIdx.x], block_aggregate);
    	//d_out[blockIdx.x] = reduction_op((T)0, block_aggregate);
}

/**
 * Utility class for dispatching the appropriately-tuned kernels for device-wide reduction
 */
template <
    typename InputIteratorT,    ///< Random-access input iterator type for reading input items \iterator
    typename OutputIteratorT,   ///< Output iterator type for recording the reduced aggregate \iterator
    typename OffsetT,           ///< Signed integer type for global offsets
    typename ReductionOpT>      ///< Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt>
struct MyDispatchSegmentedReduce :
    DeviceReducePolicy<
        typename std::iterator_traits<InputIteratorT>::value_type,
        OffsetT,
        ReductionOpT>
{
    //------------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIteratorT>::value_type T;


    //------------------------------------------------------------------------------
    // Problem state
    //------------------------------------------------------------------------------

    void                *d_temp_storage;        ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t              &temp_storage_bytes;    ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    InputIteratorT      d_in;                   ///< [in] Pointer to the input sequence of data items
    OutputIteratorT     d_out;                  ///< [out] Pointer to the output aggregate
    OffsetT             num_segments;           ///< [in] The number of segments that comprise the sorting data
    OffsetT             *d_begin_offsets;       ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
    OffsetT             *d_end_offsets;         ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
    ReductionOpT        reduction_op;           ///< [in] Binary reduction functor
    cudaStream_t        stream;                 ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                debug_synchronous;      ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    int                 ptx_version;            ///< [in] PTX version

    //------------------------------------------------------------------------------
    // Constructor
    //------------------------------------------------------------------------------

    /// Constructor
    CUB_RUNTIME_FUNCTION __forceinline__
    MyDispatchSegmentedReduce(
        void*                   d_temp_storage,
        size_t                  &temp_storage_bytes,
        InputIteratorT          d_in,
        OutputIteratorT         d_out,
        OffsetT                 num_segments,
        OffsetT                 *d_begin_offsets,
        OffsetT                 *d_end_offsets,
        ReductionOpT            reduction_op,
        cudaStream_t            stream,
        bool                    debug_synchronous,
        int                     ptx_version)
    :
        d_temp_storage(d_temp_storage),
        temp_storage_bytes(temp_storage_bytes),
        d_in(d_in),
        d_out(d_out),
        num_segments(num_segments),
        d_begin_offsets(d_begin_offsets),
        d_end_offsets(d_end_offsets),
        reduction_op(reduction_op),
        stream(stream),
        debug_synchronous(debug_synchronous),
        ptx_version(ptx_version)
    {}



    //------------------------------------------------------------------------------
    // Chained policy invocation
    //------------------------------------------------------------------------------

    /// Invocation
    template <
        typename                        ActivePolicyT,                  ///< Umbrella policy active for the target device
        typename                        DeviceSegmentedReduceKernelT>   ///< Function type of cub::DeviceSegmentedReduceKernel
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t InvokePasses(
        DeviceSegmentedReduceKernelT    segmented_reduce_kernel)        ///< [in] Kernel function pointer to parameterization of cub::DeviceSegmentedReduceKernel
    {
#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported );
#else
        cudaError error = cudaSuccess;
        do
        {
            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
            {
                temp_storage_bytes = 1;
                return cudaSuccess;
            }

            // Init kernel configuration
            KernelConfig segmented_reduce_config;
            if (CubDebug(error = segmented_reduce_config.Init<typename ActivePolicyT::SegmentedReducePolicy>(segmented_reduce_kernel))) break;

            // Log device_reduce_sweep_kernel configuration
            if (debug_synchronous) _CubLog("Invoking MySegmentedDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                num_segments,
                ActivePolicyT::SegmentedReducePolicy::BLOCK_THREADS,
                (long long) stream,
                ActivePolicyT::SegmentedReducePolicy::ITEMS_PER_THREAD,
                segmented_reduce_config.sm_occupancy);

            // Invoke DeviceReduceKernel
            segmented_reduce_kernel<<<num_segments, ActivePolicyT::SegmentedReducePolicy::BLOCK_THREADS, 0, stream>>>(
                d_in,
                d_out,
                d_begin_offsets,
                d_end_offsets,
                num_segments,
                reduction_op);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
        }
        while (0);

        return error;

#endif // CUB_RUNTIME_ENABLED

    }


    /// Invocation
    template <typename ActivePolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    cudaError_t Invoke()
    {
        typedef typename MyDispatchSegmentedReduce::MaxPolicy MaxPolicyT;

        // Force kernel code-generation in all compiler passes
        return InvokePasses<ActivePolicyT>(
        	MyDeviceSegmentedReduceWithInitKernel<MaxPolicyT, InputIteratorT, OutputIteratorT, OffsetT, ReductionOpT, T>);
    }


    //------------------------------------------------------------------------------
    // Dispatch entrypoints
    //------------------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void            *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT  d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT d_out,                              ///< [out] Pointer to the output aggregate
        int             num_segments,                       ///< [in] The number of segments that comprise the sorting data
        int             *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
        int             *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
        ReductionOpT    reduction_op,                       ///< [in] Binary reduction functor
        cudaStream_t    stream,                             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous)                  ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        typedef typename MyDispatchSegmentedReduce::MaxPolicy MaxPolicyT;

        if (num_segments <= 0)
            return cudaSuccess;

        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Create dispatch functor
            MyDispatchSegmentedReduce dispatch(
                d_temp_storage, temp_storage_bytes,
                d_in, d_out,
                num_segments, d_begin_offsets, d_end_offsets,
                reduction_op,
                stream, debug_synchronous, ptx_version);

            // Dispatch to chained policy
            if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch))) break;
        }
        while (0);

        return error;
    }
};

template <
	typename            InputIteratorT,
	typename            OutputIteratorT>
CUB_RUNTIME_FUNCTION
static cudaError_t segmentedSumWithAdd(
	void                *d_temp_storage,                    ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
	size_t              &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
	InputIteratorT      d_in,                               ///< [in] Pointer to the input sequence of data items
	OutputIteratorT     d_out,                              ///< [out] Pointer to the output aggregate
	int                 num_segments,                       ///< [in] The number of segments that comprise the sorting data
	int                 *d_begin_offsets,                   ///< [in] %Device-accessible pointer to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
	int                 *d_end_offsets,                     ///< [in] %Device-accessible pointer to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
	cudaStream_t        stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
	bool                debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
{
	typedef int OffsetT;                                                    // Signed integer type for global offsets
	typedef typename std::iterator_traits<InputIteratorT>::value_type T;    // Data element type

	return MyDispatchSegmentedReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Sum>::Dispatch(
		d_temp_storage,
		temp_storage_bytes,
		d_in,
		d_out,
		num_segments,
		d_begin_offsets,
		d_end_offsets,
		cub::Sum(),
		stream,
		debug_synchronous);
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
	CUDA_CHECK(segmentedSumWithAdd(temp_storage_d, temp_storage_bytes, x, y,  num_segments, offsets_gpu, offsets_gpu + 1, streamId, false));

//	CUDA_CHECK(cudaFree(temp_storage_d));
}



template void caffe_gpu_sum<float>(const int n, const float* x, float* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId);
template void caffe_gpu_sum<double>(const int n, const double* x, double* y, const int num_segments, int* offsets_gpu, cudaStream_t streamId);

template void caffe_gpu_sum<float>(const int n, const float* x, float* y, const int m);
template void caffe_gpu_sum<double>(const int n, const double* x, double* y, const int m);


template <typename Dtype>
void caffe_cpu_sum(const int n, const Dtype* x, Dtype* y, int m) {
	if (m <= 0)
		m = n;
	int num_segments = n/m;

	int input_offset = 0;
	for (int i = 0; i < num_segments; ++i) {

		Dtype sum_value = 0;

		for (int j = 0; j < m; ++j) {
			sum_value += x[input_offset];
			input_offset++;
		}

		y[i] += sum_value;
	}
}

template void caffe_cpu_sum(const int n, const double* x, double* y, int m);
template void caffe_cpu_sum(const int n, const float* x, float* y, int m);

}  // namespace caffe
