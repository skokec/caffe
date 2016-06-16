#ifndef CPU_ONLY
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_extra.hpp"

#include "caffe/layers/gauss_conv_layer.hpp"

#include <ctime>
#include <algorithm>

namespace caffe {

// TODO: we could speed-up with vectorized read/write

// pre-compute sigma inverse values needed in Gaussian distribution (1/sigma^2, 1/sigma^3 and 1/2*1/sigma^2)
template <typename Dtype>
__global__ void conv_gauss_precompute_sigma_kernel(const int n, Dtype* buf_sigma, Dtype* buf_sigma_square_inv, Dtype* buf_sigma_cube_inv, Dtype* buf_sigma_square_inv_half, const int sigma_lower_bound) {
  CUDA_KERNEL_LOOP(index, n) {
	  Dtype sigma_value = buf_sigma[index];

	  Dtype sigma2 = sigma_value * sigma_value;
	  Dtype sigma2_inv = 1/sigma2;

	  buf_sigma[index] = sigma_value;
	  buf_sigma_square_inv[index] = sigma2_inv;
	  buf_sigma_cube_inv[index] = 1/(sigma2 * sigma_value);
	  buf_sigma_square_inv_half[index] = (0.5 * sigma2_inv) ;
  }
}

template <typename Dtype>
__global__ void conv_gauss_distributions_kernel(const int N, const int k_w, int k_h,
												const Dtype* W, const Dtype* MU1, const Dtype* MU2, const Dtype* SIGMA_2_INV, const Dtype* SIGMA_3_INV, const Dtype* SIGMA_2_INV_HALF,
												Dtype* guass_dist, Dtype* guass_deriv_mu1, Dtype* guass_deriv_mu2, Dtype* guass_deriv_sigma) {

	const int filter_size = k_w * k_h;

	for (int n = blockIdx.z * blockDim.z + threadIdx.z; n < N; n += blockDim.z * gridDim.z){
		// read w, mu1, mu2, sigma and other data needed to compute gaussian Distributions
		//const Dtype w = W[n];
		const Dtype mu1 = MU1[n];
		const Dtype mu2 = MU2[n];
		const Dtype sigma_square_inv = SIGMA_2_INV[n];
		const Dtype sigma_square_inv_half = SIGMA_2_INV_HALF[n];
		const Dtype sigma_cube_inv = SIGMA_3_INV[n];


		// blockDim by x and y should always be 1 since whole filter will always fit into one block, so just retrive filter x,y indeces and calculate gaussians
		const int x = threadIdx.x;
		const int y = threadIdx.y;

		const Dtype dist_x = x - mu1;
		const Dtype dist_x_2 = dist_x*dist_x;

		const Dtype dist_y = y - mu2;
		const Dtype dist_y_2 = dist_y*dist_y;

		const Dtype dist = dist_x_2 + dist_y_2;
		const Dtype gauss_value = exp( -dist * sigma_square_inv_half);

		const int ptr_offset =  n * filter_size + y * k_w + x;

		guass_dist[ptr_offset] =  gauss_value;
		guass_deriv_mu1[ptr_offset] = (dist_x * sigma_square_inv) * gauss_value;
		guass_deriv_mu2[ptr_offset] = (dist_y * sigma_square_inv) * gauss_value;
		guass_deriv_sigma[ptr_offset] = (dist * sigma_cube_inv) * gauss_value;
	}
}


template <typename Dtype>
__global__ void scal_kernel_batched(const int n, const Dtype* a, const Dtype* x, Dtype* y, const int m) {

	for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < m; j += blockDim.y * gridDim.y) {
		Dtype a_value = a[j];
		for (int i = j * n + blockIdx.x * blockDim.x + threadIdx.x; i < n* (1 + j) ; i += blockDim.x * gridDim.x) {
			y[i] = a_value * x[i];
		}
	}
}


template <typename Dtype>
__global__ void axpby_kernel_batched(const int n, const Dtype a_factor, const Dtype* a, const Dtype* x, const Dtype* b, Dtype* y, const int m) {

	for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < m; j += blockDim.y * gridDim.y) {
		Dtype a_value = a[j] * a_factor;
		Dtype b_value = b[j];
		for (int i = j * n + blockIdx.x * blockDim.x + threadIdx.x; i < n * (1 + j); i += blockDim.x * gridDim.x) {
			y[i] = a_value * x[i] + b_value * y[i];
		}
	}
}

template <typename Dtype>
__global__ void add_sorted_kernel(const int S, const int G, const int F, const int n, const Dtype* unsorted_input, Dtype* sorted_output) {
	for (int f = blockIdx.z * blockDim.z + threadIdx.z; f < F; f += blockDim.z * gridDim.z) {
		for (int s = blockIdx.y * blockDim.y + threadIdx.y; s < S; s += blockDim.y * gridDim.y) {
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

				Dtype sum_g = 0;
				for (int g = 0; g < G; ++g) {
					sum_g += unsorted_input[ ((s*G + g)*F  + f )*n + i];
				}

				sorted_output[(f*S + s)*n + i] = sum_g;
			}
		}
	}
}
template <typename Dtype>
__global__ void inv_kernel(const int n, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = 1 / x[index];
  }
}

template <typename Dtype>
__global__ void mirror_kernel(const int S, const int F, const int n, const Dtype* x, Dtype* y) {

	for (int f = blockIdx.z * blockDim.z + threadIdx.z; f < F; f += blockDim.z * gridDim.z) {
		for (int s = blockIdx.y * blockDim.y + threadIdx.y; s < S; s += blockDim.y * gridDim.y) {
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
				// perform kernel mirroring by setting y[i] = x[n-i -1]
				// at the same time switch S and F indexes
				y[(s*F + f) * n + i] = x[(f*S + s) * n + n - i -1];
			}
		}
	}
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::precompute_guassian_weights_gpu(bool is_backward_pass) {

	clock_t start_t = clock();

	Dtype* weight = this->weight_buffer_->mutable_gpu_data();

// NOT implemented yet
//	Dtype* weight_vert = this->weight_vert_buffer_->mutable_gpu_data();
//	Dtype* weight_horiz = this->weight_horiz_buffer_->mutable_gpu_data();

	// pre-compute weights from guassian params
	Blob<Dtype>& gauss_param_buffer_w = *this->param_buffer_w_;
	Blob<Dtype>& gauss_param_buffer_mu1 = *this->param_buffer_mu1_;
	Blob<Dtype>& gauss_param_buffer_mu2 = *this->param_buffer_mu2_;
	Blob<Dtype>& gauss_param_buffer_sigma = *this->param_buffer_sigma_;

	Blob<Dtype>& gauss_param_buffer_sigma_square_inv = param_buffer_sigma_square_inv_;
	Blob<Dtype>& gauss_param_buffer_sigma_cube_inv = param_buffer_sigma_cube_inv_;
	Blob<Dtype>& gauss_param_buffer_sigma_square_inv_half = param_buffer_sigma_square_inv_half_;

	const Dtype* gauss_params_w = gauss_param_buffer_w.gpu_data();
	Dtype* gauss_params_mu1 = gauss_param_buffer_mu1.mutable_gpu_data();
	Dtype* gauss_params_mu2 = gauss_param_buffer_mu2.mutable_gpu_data();
	Dtype* gauss_params_sigma = gauss_param_buffer_sigma.mutable_gpu_data();

	Dtype* gauss_params_sigma_square_inv = gauss_param_buffer_sigma_square_inv.mutable_gpu_data();
	Dtype* gauss_params_sigma_cube_inv = gauss_param_buffer_sigma_cube_inv.mutable_gpu_data();
	Dtype* gauss_params_sigma_square_inv_half = gauss_param_buffer_sigma_square_inv_half.mutable_gpu_data();

	const int S = this->conv_in_channels_;
	const int F = this->conv_out_channels_;
	const int G = NUM_GAUSS;

	const int K_w = this->kernel_w_;
	const int K_h = this->kernel_h_;

	if (this->use_gmm_weight_normalization) {
		CHECK_EQ(0,1) << "GMM weight normalization not implemented with new version!!";
	}

	// clip sigma, mu1 and mu2 to within bounds
	caffe_gpu_clip_lower(gauss_param_buffer_sigma.count(), this->gmm_sigma_lower_bound, gauss_params_sigma, gauss_params_sigma);

	caffe_gpu_clip_lower(gauss_param_buffer_mu1.count(), (Dtype)this->gmm_component_border_bound, gauss_params_mu1, gauss_params_mu1);
	caffe_gpu_clip_lower(gauss_param_buffer_mu2.count(), (Dtype)this->gmm_component_border_bound, gauss_params_mu2, gauss_params_mu2);

	caffe_gpu_clip_upper(gauss_param_buffer_mu1.count(), this->kernel_w_-1 - (Dtype)this->gmm_component_border_bound, gauss_params_mu1, gauss_params_mu1);
	caffe_gpu_clip_upper(gauss_param_buffer_mu2.count(), this->kernel_h_-1 - (Dtype)this->gmm_component_border_bound, gauss_params_mu2, gauss_params_mu2);

	// 0. precompute  sigma^2, sigma^3 and (sigma^2)/2
	conv_gauss_precompute_sigma_kernel<Dtype><<<CAFFE_GET_BLOCKS(S*G*F), CAFFE_CUDA_NUM_THREADS>>>(S*G*F, gauss_params_sigma, gauss_params_sigma_square_inv, gauss_params_sigma_cube_inv, gauss_params_sigma_square_inv_half, this->gmm_sigma_lower_bound);

	/*{
		const Dtype* gauss_params_w = gauss_param_buffer_w.cpu_data();
		const Dtype* gauss_params_mu1 = gauss_param_buffer_mu1.cpu_data();
		const Dtype* gauss_params_mu2 = gauss_param_buffer_mu2.cpu_data();
		const Dtype* gauss_params_sigma = gauss_param_buffer_sigma.cpu_data();

		const Dtype* gauss_params_sigma_square_inv = gauss_param_buffer_sigma_square_inv.cpu_data();
		const Dtype* gauss_params_sigma_cube_inv = gauss_param_buffer_sigma_cube_inv.cpu_data();
		const Dtype* gauss_params_sigma_square_inv_half = gauss_param_buffer_sigma_square_inv_half.cpu_data();
	}*/

	// 1. for each pixel in [SxGxF] x [K_w x K_h] compute G (Gauss distribution), dG/dx, dG/dy, dG/dsigma

	// cuda dimension X runs over K_w, Y over K_h and dimension Z over all filters
	// we translate cuda thread X,Y dimensions directly to filter indexces of size K_w, K_h and assign cuda thread Z dimension with
	// several filters to fill as many CAFFE_CUDA_NUM_THREADS threads available (i.e. multiple filter can be processed in one cuda block)
	dim3 threadsPerBlock(K_w, K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
	dim3 numBlocks(1, 1, (S*G*F + threadsPerBlock.z - 1) / threadsPerBlock.z);

	Dtype* gauss_dist = this->guass_dist_buffer_.mutable_gpu_data();

	Dtype* deriv_weight = this->deriv_weight_buffer_->mutable_gpu_data();
	Dtype* deriv_mu1 = this->deriv_mu1_buffer_->mutable_gpu_data();
	Dtype* deriv_mu2 = this->deriv_mu2_buffer_->mutable_gpu_data();
	Dtype* deriv_sigma = this->deriv_sigma_buffer_->mutable_gpu_data();

	conv_gauss_distributions_kernel<Dtype><<<numBlocks,threadsPerBlock>>>(S*G*F, K_w, K_h, gauss_params_w, gauss_params_mu1, gauss_params_mu2, gauss_params_sigma_square_inv, gauss_params_sigma_cube_inv, gauss_params_sigma_square_inv_half, gauss_dist, deriv_mu1, deriv_mu2, deriv_sigma);

	/*{
		const Dtype* gauss_dist = this->guass_dist_buffer_.cpu_data();
		const Dtype* deriv_mu1 = this->deriv_mu1_buffer_->cpu_data();
		const Dtype* deriv_mu2 = this->deriv_mu2_buffer_->cpu_data();
		const Dtype* deriv_sigma = this->deriv_sigma_buffer_->cpu_data();
	}*/

	// 2. for each filter (G, dG/dx, dG/dy, dG/dsigma) calculate sums (use different sums if using normalization by square sum)
	Dtype* guass_norm = this->guass_norm_buffer_.mutable_gpu_data();
	Dtype* deriv_mu1_sums = this->deriv_mu1_sums_buffer_.mutable_gpu_data();
	Dtype* deriv_mu2_sums = this->deriv_mu2_sums_buffer_.mutable_gpu_data();
	Dtype* deriv_sigma_sums = this->deriv_sigma_sums_buffer_.mutable_gpu_data();

	// TODO: all three sums can be done in parallel, do we need seperate streams to make this run in parallel ?
	if (this->use_gmm_square_gauss_normalization) {
		// when using square gauss normalization derivatives dG/dx, dG/dy, dG/dsigma need to be multiplied by un-weighted, un-normalized gaussian dstirubution i.e. gauss_dist
		Dtype* deriv_mu1_times_gauss_dist = this->deriv_mu1_times_gauss_dist_buffer_.mutable_gpu_data();
		Dtype* deriv_mu2_times_gauss_dist = this->deriv_mu2_times_gauss_dist_buffer_.mutable_gpu_data();
		Dtype* deriv_sigma_times_gauss_dist = this->deriv_sigma_times_gauss_dist_buffer_.mutable_gpu_data();

		caffe_gpu_mul((S*F*G) * (K_w*K_h), gauss_dist, deriv_mu1, deriv_mu1_times_gauss_dist); // deriv_mu1_times_gauss_dist = gauss_dist * deriv_mu1;
		caffe_gpu_mul((S*F*G) * (K_w*K_h), gauss_dist, deriv_mu2, deriv_mu2_times_gauss_dist); // deriv_mu2_times_gauss_dist = gauss_dist * deriv_mu2;
		caffe_gpu_mul((S*F*G) * (K_w*K_h), gauss_dist, deriv_sigma, deriv_sigma_times_gauss_dist); // deriv_sigma_times_gauss_dist = gauss_dist * deriv_sigma;

		/*{
			const Dtype* deriv_mu1_times_gauss_dist = this->deriv_mu1_times_gauss_dist_buffer_.cpu_data();
			const Dtype* deriv_mu2_times_gauss_dist = this->deriv_mu2_times_gauss_dist_buffer_.cpu_data();
			const Dtype* deriv_sigma_times_gauss_dist = this->deriv_sigma_times_gauss_dist_buffer_.cpu_data();
		}*/

		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu1_times_gauss_dist, deriv_mu1_sums, S*F*G, this->tmp_precomp_index_gpu);
		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu2_times_gauss_dist, deriv_mu2_sums, S*F*G, this->tmp_precomp_index_gpu);
		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_sigma_times_gauss_dist, deriv_sigma_sums, S*F*G, this->tmp_precomp_index_gpu);

		caffe_gpu_scal((S*F*G), (Dtype)2, deriv_mu1_sums);
		caffe_gpu_scal((S*F*G), (Dtype)2, deriv_mu2_sums);
		caffe_gpu_scal((S*F*G), (Dtype)2, deriv_sigma_sums);
	} else {

		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu1, deriv_mu1_sums, S*F*G, this->tmp_precomp_index_gpu);
		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu2, deriv_mu2_sums, S*F*G, this->tmp_precomp_index_gpu);
		caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_sigma, deriv_sigma_sums, S*F*G, this->tmp_precomp_index_gpu);
	}

	/*{
		const Dtype* deriv_mu1_sums = this->deriv_mu1_sums_buffer_.cpu_data();
		const Dtype* deriv_mu2_sums = this->deriv_mu2_sums_buffer_.cpu_data();
		const Dtype* deriv_sigma_sums = this->deriv_sigma_sums_buffer_.cpu_data();
	}*/

	if (this->use_gmm_gauss_normalization == false) {
		// set guass_norm to 1 if we sould NOT normalize to sum of 1
		caffe_gpu_set((S*F*G), (Dtype)1, guass_norm);

	} else if (this->use_gmm_square_gauss_normalization) {
		// we need to normalize to sum of squares to 1
		Dtype* gauss_dist_square = this->gauss_dist_square_buffer_.mutable_gpu_data();

		caffe_gpu_mul((S*F*G) * (K_w*K_h), gauss_dist, gauss_dist, gauss_dist_square); // gauss_dist_square = gauss_dist * gauss_dist;
		caffe_gpu_sum((S*F*G) * (K_w*K_h), gauss_dist_square, guass_norm, S*F*G, this->tmp_precomp_index_gpu);
	} else {
		// we need to normalize to sum of 1
		caffe_gpu_sum((S*F*G) * (K_w*K_h), gauss_dist, guass_norm, S*F*G, this->tmp_precomp_index_gpu);
	}

	/*{
		const Dtype* gauss_dist_square = this->gauss_dist_square_buffer_.cpu_data();
		const Dtype* guass_norm = this->guass_norm_buffer_.cpu_data();
	}*/

	// invert guass_norm i.e. guass_norm = 1/guass_norm
	inv_kernel<Dtype><<<CAFFE_GET_BLOCKS(S*G*F), CAFFE_CUDA_NUM_THREADS>>>(S*G*F, guass_norm, guass_norm);

	/*{
		guass_norm = this->guass_norm_buffer_.mutable_gpu_data();
		const Dtype* guass_norm = this->guass_norm_buffer_.cpu_data();
	}*/

	// gauss_mu1_sum = abs(gauss_mu1_sum) > 1e-10 ? gauss_mu1_sum : 0;
	caffe_gpu_clip_eps(this->deriv_mu1_sums_buffer_.count(), (Dtype)1e-10, deriv_mu1_sums, deriv_mu1_sums);
	caffe_gpu_clip_eps(this->deriv_mu2_sums_buffer_.count(), (Dtype)1e-10, deriv_mu2_sums, deriv_mu2_sums);

	// 3. for each filter G and derivative filters dG/dx, dG/dy, dG/dsigma apply its normalization terms
	threadsPerBlock = dim3(K_w* K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
	numBlocks = dim3(1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// deriv_weight = gauss_dist * guass_norm
	scal_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, guass_norm, gauss_dist, deriv_weight, S*F*G);

	/*{
		const Dtype* deriv_weight = this->deriv_weight_buffer_->cpu_data();
	}*/


	// !after! weight and deriv_weight are computed we can add weight to guass_norm which will be used in remaining derivateives and main kernel
	caffe_gpu_mul(S*F*G, gauss_params_w, guass_norm, guass_norm); // guass_norm = gauss_params_w / guass_norm;

	/*{
		guass_norm = this->guass_norm_buffer_.mutable_gpu_data();
		const Dtype* guass_norm = this->guass_norm_buffer_.cpu_data();
	}*/

	// apply gauss normalization factors directly to sums to avoid additional call to scal_kernel_batched
	caffe_gpu_mul(S*F*G, guass_norm, deriv_mu1_sums, deriv_mu1_sums); // deriv_mu1_sums = deriv_mu1_sums * guass_norm;
	caffe_gpu_mul(S*F*G, guass_norm, deriv_mu2_sums, deriv_mu2_sums); // deriv_mu2_sums = deriv_mu2_sums * guass_norm;
	caffe_gpu_mul(S*F*G, guass_norm, deriv_sigma_sums, deriv_sigma_sums); // deriv_sigma_sums = deriv_sigma_sums * guass_norm;

	/*{
		deriv_mu1_sums = this->deriv_mu1_sums_buffer_.mutable_gpu_data();
		deriv_mu2_sums = this->deriv_mu2_sums_buffer_.mutable_gpu_data();
		deriv_sigma_sums = this->deriv_sigma_sums_buffer_.mutable_gpu_data();

		const Dtype* deriv_mu1_sums = this->deriv_mu1_sums_buffer_.cpu_data();
		const Dtype* deriv_mu2_sums = this->deriv_mu2_sums_buffer_.cpu_data();
		const Dtype* deriv_sigma_sums = this->deriv_sigma_sums_buffer_.cpu_data();
	}*/

	// create normalized derivative filters
	axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, (Dtype)-1, deriv_mu1_sums, deriv_weight,  guass_norm, deriv_mu1, S*F*G);
	axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, (Dtype)-1, deriv_mu2_sums, deriv_weight,  guass_norm, deriv_mu2, S*F*G);
	axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, (Dtype)-1, deriv_sigma_sums, deriv_weight,  guass_norm, deriv_sigma, S*F*G);

	/*{
		deriv_mu1 = this->deriv_mu1_buffer_->mutable_gpu_data();
		deriv_mu2 = this->deriv_mu2_buffer_->mutable_gpu_data();
		deriv_sigma = this->deriv_sigma_buffer_->mutable_gpu_data();
		const Dtype* deriv_mu1 = this->deriv_mu1_buffer_->cpu_data();
		const Dtype* deriv_mu2 = this->deriv_mu2_buffer_->cpu_data();
		const Dtype* deriv_sigma = this->deriv_sigma_buffer_->cpu_data();
	}*/

	// 4. calculate main kernel weights by applying gauss norm and weights, and suming over SxGxF kernels into FxS kernels (in correct order i.e. rearagning them at the same time)

	// gauss_dist = w/norm * gauss_dist (note, guass_norm should be w/norm)
	scal_kernel_batched<Dtype><<<numBlocks,threadsPerBlock>>>(K_w * K_h, guass_norm, gauss_dist, gauss_dist, S*F*G);

	/*{
		gauss_dist = this->guass_dist_buffer_.mutable_gpu_data();
		const Dtype* gauss_dist = this->guass_dist_buffer_.cpu_data();
	}*/

	threadsPerBlock = dim3(K_w*K_h, sqrt(CAFFE_CUDA_NUM_THREADS/(K_w * K_h) ), sqrt(CAFFE_CUDA_NUM_THREADS/(K_w * K_h) ) );
	numBlocks = dim3(1, (S + threadsPerBlock.y - 1) / threadsPerBlock.y, (F + threadsPerBlock.z - 1) / threadsPerBlock.z);

	add_sorted_kernel<Dtype><<<numBlocks,threadsPerBlock>>>(S, G, F, K_w*K_h, gauss_dist, weight);

	// 4. calculate seperable filters

	// 5. create error kernel for back-propagation by reversing the kernel

	Dtype* deriv_error = this->deriv_error_buffer_->mutable_gpu_data();

	threadsPerBlock = dim3(K_w*K_h, sqrt(CAFFE_CUDA_NUM_THREADS/(K_w * K_h) ), sqrt(CAFFE_CUDA_NUM_THREADS/(K_w * K_h) ) );
	numBlocks = dim3(1, (S + threadsPerBlock.y - 1) / threadsPerBlock.y, (F + threadsPerBlock.z - 1) / threadsPerBlock.z);

	mirror_kernel<Dtype><<<numBlocks,threadsPerBlock>>>(S, F, K_w*K_h, weight, deriv_error);

	cudaDeviceSynchronize();

	clock_t end_t = clock();

//	LOG(INFO) << "precompute_guassian_weights (GPU) done in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);

	/*{
		weight = this->weight_buffer_->mutable_gpu_data();
		Dtype* weight = this->weight_buffer_->mutable_cpu_data();
		Dtype* deriv_error = this->deriv_error_buffer_->mutable_cpu_data();

		Dtype* deriv_weight = this->deriv_weight_buffer_->mutable_cpu_data();
		Dtype* deriv_mu1 = this->deriv_mu1_buffer_->mutable_cpu_data();
		Dtype* deriv_mu2 = this->deriv_mu2_buffer_->mutable_cpu_data();
		Dtype* deriv_sigma = this->deriv_sigma_buffer_->mutable_cpu_data();
	}*/
}

template void GaussianConvLayer<float>::precompute_guassian_weights_gpu(bool is_backward_pass);
template void GaussianConvLayer<double>::precompute_guassian_weights_gpu(bool is_backward_pass);

}  // namespace caffe
#endif
