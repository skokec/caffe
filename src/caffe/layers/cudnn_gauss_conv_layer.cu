#ifdef USE_CUDNN
#include <vector>
#include <memory>

#include "caffe/layers/gauss_conv_layer.hpp"

#include "caffe/util/math_functions_extra.hpp"
#include "caffe/util/custom_cub.cuh"

namespace caffe {

__global__ void sync_gauss_conv_groups() { }

template <typename Dtype>
void CuDNNGaussianConvLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  //const Dtype* weight = this->weight_buffer_->gpu_data();
  //cudaDeviceSynchronize();

  clock_t start_t = clock();

  // compile guassian parameters into kernels for regular CNN
  //this->precompute_guassian_weights_gpu(true); // old version

  cudaEvent_t kernel_precomputed;
  CUDA_CHECK(cudaEventCreate(&kernel_precomputed));

  // check if we need to do merging of components;
  // make sure we check based on steps done in backpropagation and we should avoid merging if only forward is called (by default current_iteration_index=0 so start at second iter
  bool do_merginig_optmization = this->gmm_merge_iteration_step > 0 && (this->current_iteration_index + 1) % this->gmm_merge_iteration_step == 0 ? true : false;

  // if during training then merge components if needed
  if (do_merginig_optmization) {
	  merge_components();
  }

  // newer version computes kernels on-demand only if needed so we do not compile deriv kernels when testing only
  set_buffers_dirty(); // ensure we start with cleared buffers
  const Dtype* weight = this->get_weight_filters(stream_[0])->gpu_data();

  CUDA_CHECK(cudaEventRecord(kernel_precomputed, stream_[0]));

#ifndef NDEBUG
  // if we are in debug mode then calculate derivative filters as well
  LOG(INFO) << "Pre-computing derivative filters as well due to DEBUG mode enabled !!";

  shared_ptr<Blob<Dtype> > deriv_weight_kernel_buf = this->kernel_buf.deriv_weight;
  shared_ptr<Blob<Dtype> > deriv_mu1_kernel_buf = this->kernel_buf.deriv_mu1;
  shared_ptr<Blob<Dtype> > deriv_mu2_kernel_buf = this->kernel_buf.deriv_mu2;
  shared_ptr<Blob<Dtype> > deriv_sigma_kernel_buf = this->kernel_buf.deriv_sigma;

  const Dtype* deriv_weight_kernel = this->get_weight_derivative_filters(deriv_weight_kernel_buf)->gpu_data();
  const Dtype* deriv_mu1_kernel = this->get_mu1_derivative_filters(deriv_mu1_kernel_buf, deriv_weight_kernel_buf)->gpu_data();
  const Dtype* deriv_mu2_kernel = this->get_mu2_derivative_filters(deriv_mu2_kernel_buf, deriv_weight_kernel_buf)->gpu_data();
  const Dtype* deriv_sigma_kernel = this->get_sigma_derivative_filters(deriv_sigma_kernel_buf, deriv_weight_kernel_buf)->gpu_data();
#endif

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
    //	clock_t start_t = clock();
      // Filters.
      CUDA_CHECK(cudaStreamWaitEvent(stream_[g], kernel_precomputed, 0));
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));
      //clock_t end_t = clock();
      //LOG(INFO) << "cudnn forward pass in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);
      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->param_buffer_bias_->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_gauss_conv_groups<<<1, 1>>>();
  }
  CUDA_CHECK(cudaEventDestroy(kernel_precomputed));
  //cudaDeviceSynchronize();
  clock_t end_t = clock();
}

template <typename Dtype>
void CuDNNGaussianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	bool do_mean_optmization = this->gmm_mean_iteration_step > 0 && this->current_iteration_index % this->gmm_mean_iteration_step == 0 ? true : false;
	bool do_sigma_optmization = this->gmm_sigma_iteration_step > 0 && this->current_iteration_index % this->gmm_sigma_iteration_step == 0 ? true : false;

	this->current_iteration_index++;

	// param_<XYZ> are of size [1 x S x G x F]
	Dtype* param_w_diff = this->param_buffer_w_->mutable_gpu_diff();
	Dtype* param_mu1_diff = this->param_buffer_mu1_->mutable_gpu_diff();
	Dtype* param_mu2_diff = this->param_buffer_mu2_->mutable_gpu_diff();
	Dtype* param_sigma_diff = this->param_buffer_sigma_->mutable_gpu_diff();

	if (this->param_propagate_down_[0]) {
		caffe_gpu_set_async(this->param_buffer_w_->count(), (Dtype)0, param_w_diff, paralel_streams[0]);
		caffe_gpu_set_async(this->param_buffer_mu1_->count(), (Dtype)0, param_mu1_diff, paralel_streams[1]);
		caffe_gpu_set_async(this->param_buffer_mu2_->count(), (Dtype)0, param_mu2_diff, paralel_streams[2]);
		caffe_gpu_set_async(this->param_buffer_sigma_->count(), (Dtype)0, param_sigma_diff, paralel_streams[3]);
	}

	// ensure all work from previous kernel and from caffe_gpu_set_async is done
	cudaDeviceSynchronize();

	const int I = this->num_;
	const int S = this->conv_in_channels_;
	const int F = this->conv_out_channels_;
	const int G = this->NUM_GAUSS;

	//const int K_w = this->kernel_w_;
	//const int K_h = this->kernel_h_;

	clock_t start_t = clock();

	// Gradient w.r.t. bias.
	if (this->bias_term_ && this->param_propagate_down_[1]) {

		Dtype* bias_diff = this->param_buffer_bias_->mutable_gpu_diff();

		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();

			int g = 0;
			CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
					cudnn::dataType<Dtype>::one,
					top_descs_[i],  top_diff + top_offset_ * g,
					cudnn::dataType<Dtype>::one,
					bias_desc_, bias_diff + bias_offset_ * g));
		}
	}

	cudaEvent_t weight_kernel_precomputed, weight_deriv_kernel_precomputed, mu1_deriv_kernel_precomputed, mu2_deriv_kernel_precomputed, sigma_deriv_kernel_precomputed;
	CUDA_CHECK(cudaEventCreate(&weight_kernel_precomputed));
	CUDA_CHECK(cudaEventCreate(&weight_deriv_kernel_precomputed));
	CUDA_CHECK(cudaEventCreate(&mu1_deriv_kernel_precomputed));
	CUDA_CHECK(cudaEventCreate(&mu2_deriv_kernel_precomputed));
	CUDA_CHECK(cudaEventCreate(&sigma_deriv_kernel_precomputed));

	vector<cudaEvent_t> error_backprop_done;

	//for (int g = 0; g < this->group_; g++)
	{
		error_backprop_done.push_back(cudaEvent_t());
		CUDA_CHECK(cudaEventCreate(&error_backprop_done.back()));
	}

	// Gradient w.r.t. bottom data.
	// result from get_weight_filters, get_mu1_derivative_filters, get_mu2_derivative_filters and get_sigma_derivative_filters uses shared output buffer
	// so calc gradient for all data before moving to next derivative
	const Dtype* weight_kernel = this->get_weight_filters(stream_[1*this->group_ + 0])->gpu_data();

	CUDA_CHECK(cudaEventRecord(weight_kernel_precomputed, stream_[1*this->group_ + 0]));

	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->gpu_diff();
		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

		if (propagate_down[i]) {
			int g = 0;
			CUDA_CHECK(cudaStreamWaitEvent(stream_[1*this->group_ + g], weight_kernel_precomputed, 0));
			CUDNN_CHECK(cudnnConvolutionBackwardData(
					handle_[1*this->group_ + g],
					cudnn::dataType<Dtype>::one,
					filter_desc_, weight_kernel + this->weight_offset_ * g,
					top_descs_[i], top_diff + top_offset_ * g,
					conv_descs_[i],
					bwd_data_algo_[i], workspace[1*this->group_ + g],
					workspace_bwd_data_sizes_[i],
					cudnn::dataType<Dtype>::zero,
					bottom_descs_[i], bottom_diff + bottom_offset_ * g));
		}
	}
	//for (int g = 0; g < this->group_; g++)
	{
		int g = 0;
		CUDA_CHECK(cudaEventRecord(error_backprop_done[g], stream_[1*this->group_ + g]));
	}

	// Gradient w.r.t. parameters.
	if (this->param_propagate_down_[0]) {

		// Gradient w.r.t. weights.
		// result from get_weight_filters, get_mu1_derivative_filters, get_mu2_derivative_filters and get_sigma_derivative_filters share the output buffer
		// so calc derivative weights for all data before moving to next derivative
		shared_ptr<Blob<Dtype> > deriv_weight_kernel_buf = this->kernel_buf.deriv_weight; // paralel_streams[0]
#ifndef NDEBUG
		shared_ptr<Blob<Dtype> > deriv_mu1_kernel_buf = this->kernel_buf.deriv_mu1;
		shared_ptr<Blob<Dtype> > deriv_mu2_kernel_buf = this->kernel_buf.deriv_mu2;
		shared_ptr<Blob<Dtype> > deriv_sigma_kernel_buf = this->kernel_buf.deriv_sigma;
#else
		shared_ptr<Blob<Dtype> > deriv_mu1_kernel_buf = this->kernel_buf.weights;  		 // paralel_streams[1]
		shared_ptr<Blob<Dtype> > deriv_mu2_kernel_buf = this->kernel_buf.deriv_mu2;		 // paralel_streams[2]
		shared_ptr<Blob<Dtype> > deriv_sigma_kernel_buf = this->kernel_buf.weights;		 // paralel_streams[1]
#endif

		CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[0], weight_kernel_precomputed, 0));
		const Dtype* deriv_weight_kernel = this->get_weight_derivative_filters(deriv_weight_kernel_buf, paralel_streams[0])->gpu_data();
		//CUDA_CHECK(cudaEventRecord(weight_deriv_kernel_precomputed, paralel_streams[0]));

		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();
			const Dtype* bottom_data = bottom[i]->gpu_data();

			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

			//CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[0], weight_deriv_kernel_precomputed, 0));
			filterActs_YxX_color(bottom_data, top_diff, deriv_weight_kernel, param_w_diff,
					I, S, F, G, 0, 0, 0,
					this->conv_in_width_, this->conv_in_height_,
					this->width_out_, this->height_out_,
					this->kernel_w_, this->kernel_h_,
					this->pad_w_, 0, paralel_streams[0]);
		}

		// before we can reuse weight buffer we need to ensure it is not needed any more
		for (int g = 0; g < error_backprop_done.size(); ++g){
			CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[1], error_backprop_done[g], 0));
		}
		// Gradient w.r.t. means (mu1 and mu2).
		// result from get_weight_filters, get_mu1_derivative_filters, get_mu2_derivative_filters and get_sigma_derivative_filters share the output buffer
		// so calc derivative of means for all data before moving to next derivative
		if (do_mean_optmization) {
			CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[1], weight_deriv_kernel_precomputed, 0));
			const Dtype* deriv_mu1_kernel = this->get_mu1_derivative_filters(deriv_mu1_kernel_buf, deriv_weight_kernel_buf, paralel_streams[1])->gpu_data();
			//CUDA_CHECK(cudaEventRecord(mu2_deriv_kernel_precomputed, paralel_streams[1]));

			for (int i = 0; i < top.size(); ++i) {
				const Dtype* top_diff = top[i]->gpu_diff();
				const Dtype* bottom_data = bottom[i]->gpu_data();

				//CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[1], mu1_deriv_kernel_precomputed, 0));
				// Backward through cuDNN in parallel over groups and gradients.
				filterActs_YxX_color(bottom_data, top_diff, deriv_mu1_kernel, param_mu1_diff,
						I, S, F, G, 0, 0, 0,
						this->conv_in_width_, this->conv_in_height_,
						this->width_out_, this->height_out_,
						this->kernel_w_, this->kernel_h_,
						this->pad_w_, 0, paralel_streams[1]);
			}

			CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[2], weight_deriv_kernel_precomputed, 0));
			const Dtype* deriv_mu2_kernel = this->get_mu2_derivative_filters(deriv_mu2_kernel_buf, deriv_weight_kernel_buf, paralel_streams[2])->gpu_data();
			//CUDA_CHECK(cudaEventRecord(mu2_deriv_kernel_precomputed, paralel_streams[2]));

			for (int i = 0; i < top.size(); ++i) {
				const Dtype* top_diff = top[i]->gpu_diff();
				const Dtype* bottom_data = bottom[i]->gpu_data();

				//CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[2], mu2_deriv_kernel_precomputed, 0));
				filterActs_YxX_color(bottom_data, top_diff, deriv_mu2_kernel, param_mu2_diff,
						I, S, F, G, 0, 0, 0,
						this->conv_in_width_, this->conv_in_height_,
						this->width_out_, this->height_out_,
						this->kernel_w_, this->kernel_h_,
						this->pad_w_, 0, paralel_streams[2]);
			}
		}
		// Gradient w.r.t. variance (sigma).
		// result from get_weight_filters, get_mu1_derivative_filters, get_mu2_derivative_filters and get_sigma_derivative_filters share the output buffer
		// so calc derivative of sigma for all data before moving to next derivative
		CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[1], weight_deriv_kernel_precomputed, 0));
		const Dtype* deriv_sigma_kernel = this->get_sigma_derivative_filters(deriv_sigma_kernel_buf, deriv_weight_kernel_buf, paralel_streams[1])->gpu_data();
		//CUDA_CHECK(cudaEventRecord(mu2_deriv_kernel_precomputed, paralel_streams[1]));

		if (do_sigma_optmization) {
			for (int i = 0; i < top.size(); ++i) {
				const Dtype* top_diff = top[i]->gpu_diff();
				const Dtype* bottom_data = bottom[i]->gpu_data();

				//CUDA_CHECK(cudaStreamWaitEvent(paralel_streams[1], sigma_deriv_kernel_precomputed, 0));
				filterActs_YxX_color(bottom_data, top_diff, deriv_sigma_kernel, param_sigma_diff,
						I, S, F, G, 0, 0, 0,
						this->conv_in_width_, this->conv_in_height_,
						this->width_out_, this->height_out_,
						this->kernel_w_, this->kernel_h_,
						this->pad_w_, 0, paralel_streams[1]);


			}

		}
	}
	sync_gauss_conv_groups<<<1, 1>>>();
	//cudaDeviceSynchronize();
	// Synchronize the work across groups, each of which went into its own
	// stream, by launching an empty kernel into the default (null) stream.
	// NOLINT_NEXT_LINE(whitespace/operators)

	CUDA_CHECK(cudaEventDestroy(weight_kernel_precomputed));
	CUDA_CHECK(cudaEventDestroy(weight_deriv_kernel_precomputed));
	CUDA_CHECK(cudaEventDestroy(mu1_deriv_kernel_precomputed));
	CUDA_CHECK(cudaEventDestroy(mu2_deriv_kernel_precomputed));
	CUDA_CHECK(cudaEventDestroy(sigma_deriv_kernel_precomputed));

	for (int g = 0; g < error_backprop_done.size(); ++g)
		CUDA_CHECK(cudaEventDestroy(error_backprop_done[g]));

	clock_t end_t = clock();
	//LOG(INFO) << "old-cudnn backward pass in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);

	// make sure to set all buffers to dirty so they will be re-computed next time they are accessed
	set_buffers_dirty();
}

// pre-compute sigma inverse values needed in Gaussian distribution (1/sigma^2, 1/sigma^3 and 1/2*1/sigma^2)
template <typename Dtype>
__global__ void conv_gauss_precompute_sigma_kernel(const int n, Dtype* buf_sigma, Dtype* buf_sigma_square_inv, Dtype* buf_sigma_cube_inv, Dtype* buf_sigma_square_inv_half, const int sigma_lower_bound) {
  CUDA_KERNEL_LOOP(index, n) {
	  Dtype sigma_value = buf_sigma[index];

	  Dtype sigma2 = sigma_value * sigma_value;
	  Dtype sigma2_inv = 1/sigma2;

	  buf_sigma_square_inv[index] = sigma2_inv;
	  buf_sigma_cube_inv[index] = 1/(sigma2 * sigma_value);
	  buf_sigma_square_inv_half[index] = (0.5 * sigma2_inv) ;
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
__global__ void conv_gauss_distributions_kernel(const int N, const int k_w, int k_h,
												const Dtype* W, const Dtype* MU1, const Dtype* MU2, const Dtype* SIGMA_2_INV_HALF,
												Dtype* guass_dist) {

	const int filter_size = k_w * k_h;

	for (int n = blockIdx.z * blockDim.z + threadIdx.z; n < N; n += blockDim.z * gridDim.z){
		// blockDim by x and y should always be 1 since whole filter will always fit into one block, so just retrive filter x,y indeces and calculate gaussians
		const int x = threadIdx.x;
		const int y = threadIdx.y;

		// read w, mu1, mu2, sigma and other data needed to compute gaussian Distributions
		//const Dtype w = W[n];
		const Dtype mu1 = MU1[n];
		const Dtype mu2 = MU2[n];
		const Dtype sigma_square_inv_half = SIGMA_2_INV_HALF[n];

		const int ptr_offset =  n * filter_size + y * k_w + x;

		const Dtype dist_x = x - mu1;
		const Dtype dist_x_2 = dist_x*dist_x;

		const Dtype dist_y = y - mu2;
		const Dtype dist_y_2 = dist_y*dist_y;

		const Dtype dist = dist_x_2 + dist_y_2;
		const Dtype gauss_value = exp( -dist * sigma_square_inv_half);

		guass_dist[ptr_offset] =  gauss_value;
	}
}


template <typename Dtype>
shared_ptr<Blob<Dtype> > CuDNNGaussianConvLayer<Dtype>::get_gauss_distribution_buffer(cudaStream_t streamId) {

	// input buffers:
	//  - this->param_buffer_w_							[S * G * F]
	//  - this->param_buffer_mu1_ (modified for bound)  [S * G * F]
	//  - this->param_buffer_mu2_ (modified for bound)	[S * G * F]
	//  - this->param_buffer_sigma_	(modified for bound)[S * G * F]

	// output buffers:
	//  - this->guass_dist_buffer_ (return)				[S * G * F * K_h * K_w]
	//  - this->param_buffer_sigma_square_inv_			[S * G * F]
	//  - this->param_buffer_sigma_cube_inv_			[S * G * F]
	//  - this->param_buffer_sigma_square_inv_half_		[S * G * F]

	if (this->dirty_gauss_dist_buffer) {
		const int S = this->conv_in_channels_;
		const int F = this->conv_out_channels_;
		const int G = this->NUM_GAUSS;

		const int K_w = this->kernel_w_;
		const int K_h = this->kernel_h_;

		if (this->use_gmm_weight_normalization) {
			CHECK_EQ(0,1) << "GMM weight normalization not implemented with new version!!";
		}

		const Dtype* gauss_params_w = this->param_buffer_w_->gpu_data();
		Dtype* gauss_params_mu1 = this->param_buffer_mu1_->mutable_gpu_data();
		Dtype* gauss_params_mu2 = this->param_buffer_mu2_->mutable_gpu_data();
		Dtype* gauss_params_sigma = this->param_buffer_sigma_->mutable_gpu_data();

		Dtype* gauss_params_sigma_square_inv = this->tmp_buf.sigma_square_inv->mutable_gpu_data();
		Dtype* gauss_params_sigma_cube_inv = this->tmp_buf.sigma_cube_inv->mutable_gpu_data();
		Dtype* gauss_params_sigma_square_inv_half = this->tmp_buf.sigma_square_inv_half->mutable_gpu_data();


		Dtype* gauss_dist = this->tmp_buf.distribution->mutable_gpu_data();

		caffe_gpu_clip_lower(this->param_buffer_sigma_->count(), this->gmm_sigma_lower_bound, gauss_params_sigma, gauss_params_sigma, streamId);

		caffe_gpu_clip_lower(this->param_buffer_mu1_->count(), (Dtype)this->gmm_component_border_bound, gauss_params_mu1, gauss_params_mu1, streamId);
		caffe_gpu_clip_lower(this->param_buffer_mu2_->count(), (Dtype)this->gmm_component_border_bound, gauss_params_mu2, gauss_params_mu2, streamId);

		caffe_gpu_clip_upper(this->param_buffer_mu1_->count(), this->kernel_w_-1 - (Dtype)this->gmm_component_border_bound, gauss_params_mu1, gauss_params_mu1, streamId);
		caffe_gpu_clip_upper(this->param_buffer_mu2_->count(), this->kernel_h_-1 - (Dtype)this->gmm_component_border_bound, gauss_params_mu2, gauss_params_mu2, streamId);

		// precompute  sigma^2, sigma^3 and (sigma^2)/2
		conv_gauss_precompute_sigma_kernel<Dtype><<<CAFFE_GET_BLOCKS(S*G*F), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(S*G*F, gauss_params_sigma, gauss_params_sigma_square_inv, gauss_params_sigma_cube_inv, gauss_params_sigma_square_inv_half, this->gmm_sigma_lower_bound);

		// calulate distribution and its normalization values
		dim3 threadsPerBlock(K_w, K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
		dim3 numBlocks(1, 1, (S*G*F + threadsPerBlock.z - 1) / threadsPerBlock.z);
		conv_gauss_distributions_kernel<Dtype><<<numBlocks,threadsPerBlock, 0, streamId>>>(S*G*F, K_w, K_h, gauss_params_w, gauss_params_mu1, gauss_params_mu2, gauss_params_sigma_square_inv_half, gauss_dist);

	}

	this->dirty_gauss_dist_buffer = false;

	//return shared_ptr<Blob<Dtype> >(shared_ptr<Blob<Dtype> >(),&this->guass_dist_buffer_);
	return this->tmp_buf.distribution;
}


template <typename Dtype>
__global__ void inv_kernel(const int n, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = 1 / x[index];
  }
}
template <typename Dtype>
shared_ptr<Blob<Dtype> > CuDNNGaussianConvLayer<Dtype>::get_gauss_normalization_buffer(cudaStream_t streamId) {

	// input buffers:
	//  - this->guass_dist_buffer_ (getter)				[S * G * F * K_h * K_w]

	// output buffers:
	//  - this->guass_norm_buffer_ (return)				[S * G * F]

	if (this->dirty_norm_buffer) {
		const int S = this->conv_in_channels_;
		const int F = this->conv_out_channels_;
		const int G = this->NUM_GAUSS;

		const int K_w = this->kernel_w_;
		const int K_h = this->kernel_h_;

		// get distribution buffer
		const Dtype* gauss_dist = this->get_gauss_distribution_buffer()->gpu_data();
		Dtype* guass_norm = this->tmp_buf.norm->mutable_gpu_data();

		if (this->use_gmm_gauss_normalization == false) {
			// set guass_norm to 1 if we sould NOT normalize to sum of 1
			caffe_gpu_set_async((S*F*G), (Dtype)1, guass_norm, streamId);

		} else if (this->use_gmm_square_gauss_normalization) {
			// we need to normalize to sum of squares to 1
			caffe_gpu_dot_batched((S*F*G) * (K_w*K_h), gauss_dist, gauss_dist, guass_norm, S*F*G, this->tmp_precomp_index_gpu, streamId);
		} else {
			// we need to normalize to sum of 1
			caffe_gpu_sum((S*F*G) * (K_w*K_h), gauss_dist, guass_norm, S*F*G, this->tmp_precomp_index_gpu, streamId);
		}

		// invert guass_norm i.e. guass_norm = 1/guass_norm
		inv_kernel<Dtype><<<CAFFE_GET_BLOCKS(S*G*F), CAFFE_CUDA_NUM_THREADS, 0, streamId>>>(S*G*F, guass_norm, guass_norm);
	}

	this->dirty_norm_buffer = false;

	//return shared_ptr<Blob<Dtype> >(shared_ptr<Blob<Dtype> >(),&this->guass_norm_buffer_);
	return this->tmp_buf.norm;

}

template <typename Dtype>
shared_ptr<Blob<Dtype> > CuDNNGaussianConvLayer<Dtype>::get_gauss_normalization_with_weight_buffer(cudaStream_t streamId) {

	// input buffers:
	//  - this->param_buffer_w_							[S * G * F]
	//  - this->guass_norm_buffer_ (getter)				[S * G * F]

	// output buffers:
	//  - this->guass_norm_with_w_buffer_ (return)		[S * G * F]

	if (this->dirty_norm_with_w_buffer) {
		const int S = this->conv_in_channels_;
		const int F = this->conv_out_channels_;
		const int G = this->NUM_GAUSS;

		// use get_gauss_distribution_buffer function to compute normalization buffer and just return it
		const Dtype* gauss_params_w = this->param_buffer_w_->gpu_data();
		const Dtype* guass_norm = this->get_gauss_normalization_buffer()->gpu_data();

		Dtype* guass_norm_with_w = this->tmp_buf.norm_with_w->mutable_gpu_data();

		// use caffe_gpu_mul_batched instead of caffe_gpu_mul to use manually defined cuda stream
		caffe_gpu_mul_batched(S*F*G, gauss_params_w, guass_norm, guass_norm_with_w, 0, streamId); // guass_norm_with_w = gauss_params_w * guass_norm; (where guass_norm is already 1/guass_norm)
	}

	this->dirty_norm_with_w_buffer = false;

	//return shared_ptr<Blob<Dtype> >(shared_ptr<Blob<Dtype> >(),&this->guass_norm_with_w_buffer_);
	return this->tmp_buf.norm_with_w;
}

template <typename Dtype>
__global__ void add_sorted_kernel(const int S, const int G, const int F, const int n, const Dtype* factor, const Dtype* unsorted_input, Dtype* sorted_output) {
	for (int f = blockIdx.z * blockDim.z + threadIdx.z; f < F; f += blockDim.z * gridDim.z) {
		for (int s = blockIdx.y * blockDim.y + threadIdx.y; s < S; s += blockDim.y * gridDim.y) {
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {

				Dtype sum_g = 0;
				for (int g = 0; g < G; ++g) {
					Dtype fac = factor[(s*G + g)*F  + f];
					sum_g += fac * unsorted_input[ ((s*G + g)*F  + f )*n + i];
				}

				sorted_output[(f*S + s)*n + i] = sum_g;
			}
		}
	}
}

template <typename Dtype>
shared_ptr<Blob<Dtype> > CuDNNGaussianConvLayer<Dtype>::get_weight_filters(cudaStream_t streamId) {

	// input buffers:
	//  - this->guass_dist_buffer_ (getter)				[S * G * F * K_h * K_w]
	//  - this->guass_norm_with_w_buffer_ (getter)		[S * G * F]

	// output buffers:
	//  - this->weight_buffer_ (return)					[S * F * K_h * K_w]



	if (this->dirty_weight_buffer) {
		const int S = this->conv_in_channels_;
		const int F = this->conv_out_channels_;
		const int G = this->NUM_GAUSS;

		const int K_w = this->kernel_w_;
		const int K_h = this->kernel_h_;

		const Dtype* gauss_dist = this->get_gauss_distribution_buffer()->gpu_data();
		const Dtype* guass_norm = this->get_gauss_normalization_with_weight_buffer()->gpu_data();

		Dtype* weight = this->kernel_buf.weights->mutable_gpu_data();

		dim3 threadsPerBlock(K_w*K_h, sqrt(CAFFE_CUDA_NUM_THREADS/(K_w * K_h) ), sqrt(CAFFE_CUDA_NUM_THREADS/(K_w * K_h) ) );
		dim3 numBlocks(1, (S + threadsPerBlock.y - 1) / threadsPerBlock.y, (F + threadsPerBlock.z - 1) / threadsPerBlock.z);

		add_sorted_kernel<Dtype><<<numBlocks,threadsPerBlock, 0, streamId>>>(S, G, F, K_w*K_h, guass_norm, gauss_dist, weight);
	}

	this->dirty_weight_buffer = false;

	return this->kernel_buf.weights;
}

template <typename Dtype>
shared_ptr<Blob<Dtype> > CuDNNGaussianConvLayer<Dtype>::get_weight_derivative_filters(shared_ptr<Blob<Dtype> > output_buffer, cudaStream_t streamId) {

	// input buffers:
	//  - this->guass_dist_buffer_ (getter)				[S * G * F * K_h * K_w]
	//  - this->guass_norm_buffer_ (getter)				[S * G * F]

	// output buffers:
	//  - this->deriv_weight_buffer_ (return)			[S * G * F * K_h * K_w]

	if (this->dirty_weight_deriv_buffer) {
		// get gauss distribution and normalization buffer
		const Dtype* gauss_dist = this->get_gauss_distribution_buffer()->gpu_data();
		const Dtype* guass_norm = this->get_gauss_normalization_buffer()->gpu_data();

		Dtype* deriv_weight = output_buffer->mutable_gpu_data();

		const int S = this->conv_in_channels_;
		const int F = this->conv_out_channels_;
		const int G = this->NUM_GAUSS;

		const int K_w = this->kernel_w_;
		const int K_h = this->kernel_h_;

		// compute weights deriv from retrieved buffers
		dim3 threadsPerBlock = dim3(K_w* K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
		dim3 numBlocks = dim3(1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// deriv_weight = gauss_dist * guass_norm
		scal_kernel_batched<Dtype><<<numBlocks,threadsPerBlock,0, streamId>>>(K_w * K_h, guass_norm, gauss_dist, deriv_weight, S*F*G);
	}

	this->dirty_weight_deriv_buffer = false;

	//return this->deriv_weight_buffer_;
	return output_buffer;
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
__global__ void conv_gauss_mu1_deriv_kernel(const int N, const int k_w, int k_h,
												const Dtype* MU1, const Dtype* SIGMA_2_INV, const Dtype* guass_dist,
												Dtype* guass_deriv_mu1) {

	const int filter_size = k_w * k_h;

	for (int n = blockIdx.z * blockDim.z + threadIdx.z; n < N; n += blockDim.z * gridDim.z){

		// blockDim by x and y should always be 1 since whole filter will always fit into one block, so just retrive filter x,y indeces and calculate gaussians
		const int x = threadIdx.x;
		const int y = threadIdx.y;

		const int ptr_offset =  n * filter_size + y * k_w + x;

		// read w, mu1, mu2, sigma and other data needed to compute gaussian Distributions
		const Dtype mu1 = MU1[n];
		const Dtype sigma_square_inv = SIGMA_2_INV[n];

		const Dtype gauss_value = guass_dist[ptr_offset];

		const Dtype dist_x = x - mu1;

		guass_deriv_mu1[ptr_offset] = (dist_x * sigma_square_inv) * gauss_value;
	}
}

#define SUM_MIN_BOUND (Dtype)1e-10

template <typename Dtype>
shared_ptr<Blob<Dtype> > CuDNNGaussianConvLayer<Dtype>::get_mu1_derivative_filters(shared_ptr<Blob<Dtype> > output_buffer, shared_ptr<Blob<Dtype> > deriv_weight_buffer, cudaStream_t streamId) {
	// input buffers:
	//  - this->deriv_weight_buffer_ (return)			[S * G * F * K_h * K_w]
	//  - this->guass_dist_buffer_ (getter)				[S * G * F * K_h * K_w]
	//  - this->guass_norm_with_w_buffer_ (getter)		[S * G * F]
	//  - this->param_buffer_mu1_ 						[S * G * F]
	//  - this->param_buffer_sigma_square_inv_			[S * G * F]

	// temp buffers:
	//  - this->deriv_mu1_sums_buffer_					[S * G * F] (shared)

	// output buffers:
	//  - this->deriv_mu1_buffer_ (return)				[S * G * F * K_h * K_w] (shared)

	if (this->dirty_mu1_deriv_buffer) {
		// get needed buffers
		const Dtype* gauss_dist = this->get_gauss_distribution_buffer()->gpu_data();
		const Dtype* gauss_norm_with_w = this->get_gauss_normalization_with_weight_buffer()->gpu_data();
		//const Dtype* deriv_weight = this->get_weight_derivative_filters(this->kernel)->gpu_data();
		const Dtype* deriv_weight = deriv_weight_buffer->gpu_data();

		const Dtype* gauss_params_sigma_square_inv = this->tmp_buf.sigma_square_inv->gpu_data(); // NOTE: this requires call to get_gauss_distribution_buffer() beforehand
		const Dtype* gauss_params_mu1 = this->param_buffer_mu1_->gpu_data();

		// temporary buffer
		Dtype* deriv_mu1_sums = this->tmp_buf.deriv_mu1_sums->mutable_gpu_data();

		// output buffer
		Dtype* deriv_mu1 = output_buffer->mutable_gpu_data();

		const int S = this->conv_in_channels_;
		const int F = this->conv_out_channels_;
		const int G = this->NUM_GAUSS;

		const int K_w = this->kernel_w_;
		const int K_h = this->kernel_h_;

		// compute weights deriv from retrieved buffers
		dim3 threadsPerBlock = dim3(K_w, K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
		dim3 numBlocks = dim3(1, 1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// computer derivative kernel
		conv_gauss_mu1_deriv_kernel<Dtype><<<numBlocks,threadsPerBlock,0, streamId>>>(S*G*F, K_w, K_h, gauss_params_mu1, gauss_params_sigma_square_inv, gauss_dist, deriv_mu1);

		// compute sum of derivative with
		if (this->use_gmm_gauss_normalization == false) {
			// if there is no normalization then there should be no derivative of normalization
			caffe_gpu_set_async((S*F*G), (Dtype)0, deriv_mu1_sums, streamId);

		} else if (this->use_gmm_square_gauss_normalization) {

			// deriv_mu1_sums = 2 * sum(gauss_dist * deriv_mu1);
			caffe_gpu_dot_batched((S*F*G) * (K_w*K_h), gauss_dist, deriv_mu1, deriv_mu1_sums, S*F*G, this->tmp_precomp_index_gpu, false, streamId);

			CUDNN_CALL_WITH_STREAM(streamId,
					caffe_gpu_scal((S*F*G), (Dtype)2, deriv_mu1_sums)
			);
		} else {
			// deriv_mu1_sums = sum(deriv_mu1);
			caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu1, deriv_mu1_sums, S*F*G, this->tmp_precomp_index_gpu, streamId);
		}
		// gauss_mu1_sum = abs(gauss_mu1_sum) > 1e-10 ? gauss_mu1_sum : 0;
		caffe_gpu_clip_eps(this->deriv_mu1_sums_buffer_.count(), SUM_MIN_BOUND, deriv_mu1_sums, deriv_mu1_sums, streamId);

		// use caffe_gpu_mul_batched instead of caffe_gpu_mul to use manually defined cuda stream
		caffe_gpu_mul_batched(S*F*G, gauss_norm_with_w, deriv_mu1_sums, deriv_mu1_sums, 0, streamId); // deriv_mu1_sums = deriv_mu1_sums * guass_norm;

		// and add normalizaion (i.e., deriv_mu1 = deriv_mu1 - deriv_weight * deriv_mu1_sums)
		threadsPerBlock = dim3(K_w* K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
		numBlocks = dim3(1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

		axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock,0,streamId>>>(K_w * K_h, (Dtype)-1, deriv_mu1_sums, deriv_weight,  gauss_norm_with_w, deriv_mu1, S*F*G);
	}

	this->dirty_mu1_deriv_buffer = false;

	return output_buffer;
	//return this->deriv_mu1_buffer_;
}


template <typename Dtype>
__global__ void conv_gauss_mu2_deriv_kernel(const int N, const int k_w, int k_h,
												const Dtype* MU2, const Dtype* SIGMA_2_INV, const Dtype* guass_dist,
												Dtype* guass_deriv_mu2) {

	const int filter_size = k_w * k_h;

	for (int n = blockIdx.z * blockDim.z + threadIdx.z; n < N; n += blockDim.z * gridDim.z){

		// blockDim by x and y should always be 1 since whole filter will always fit into one block, so just retrive filter x,y indeces and calculate gaussians
		const int x = threadIdx.x;
		const int y = threadIdx.y;

		const int ptr_offset =  n * filter_size + y * k_w + x;

		// read w, mu1, mu2, sigma and other data needed to compute gaussian Distributions
		const Dtype mu2 = MU2[n];
		const Dtype sigma_square_inv = SIGMA_2_INV[n];

		const Dtype gauss_value = guass_dist[ptr_offset];

		const Dtype dist_y = y - mu2;

		guass_deriv_mu2[ptr_offset] = (dist_y * sigma_square_inv) * gauss_value;
	}
}

template <typename Dtype>
shared_ptr<Blob<Dtype> > CuDNNGaussianConvLayer<Dtype>::get_mu2_derivative_filters(shared_ptr<Blob<Dtype> > output_buffer, shared_ptr<Blob<Dtype> > deriv_weight_buffer, cudaStream_t streamId) {
	// input buffers:
	//  - this->deriv_weight_buffer_ (return)			[S * G * F * K_h * K_w]
	//  - this->guass_dist_buffer_ (getter)				[S * G * F * K_h * K_w]
	//  - this->guass_norm_with_w_buffer_ (getter)		[S * G * F]
	//  - this->param_buffer_mu2_ 						[S * G * F]
	//  - this->param_buffer_sigma_square_inv_			[S * G * F]

	// temp buffers:
	//  - this->deriv_mu2_sums_buffer_					[S * G * F] (shared)

	// output buffers:
	//  - this->deriv_mu2_buffer_ (return)				[S * G * F * K_h * K_w] (shared)

	if (this->dirty_mu2_deriv_buffer) {
		// get needed buffers
		const Dtype* gauss_dist = this->get_gauss_distribution_buffer()->gpu_data();
		const Dtype* gauss_norm_with_w = this->get_gauss_normalization_with_weight_buffer()->gpu_data();
		//const Dtype* deriv_weight = this->get_weight_derivative_filters(this->deriv_weight_buffer_)->gpu_data();
		const Dtype* deriv_weight = deriv_weight_buffer->gpu_data();

		const Dtype* gauss_params_sigma_square_inv = this->tmp_buf.sigma_square_inv->gpu_data(); // NOTE: this requires call to get_gauss_distribution_buffer() beforehand
		const Dtype* gauss_params_mu2 = this->param_buffer_mu2_->gpu_data();

		Dtype* deriv_mu2_sums = this->tmp_buf.deriv_mu2_sums->mutable_gpu_data();

		Dtype* deriv_mu2 = output_buffer->mutable_gpu_data();

		const int S = this->conv_in_channels_;
		const int F = this->conv_out_channels_;
		const int G = this->NUM_GAUSS;

		const int K_w = this->kernel_w_;
		const int K_h = this->kernel_h_;

		// computer derivative kernel
		dim3 threadsPerBlock = dim3(K_w, K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
		dim3 numBlocks = dim3(1, 1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

		conv_gauss_mu2_deriv_kernel<Dtype><<<numBlocks,threadsPerBlock,0,streamId>>>(S*G*F, K_w, K_h, gauss_params_mu2, gauss_params_sigma_square_inv, gauss_dist, deriv_mu2);

		// compute sum of derivative with
		if (this->use_gmm_gauss_normalization == false) {
			// if there is no normalization then there should be no derivative of normalization
			caffe_gpu_set_async((S*F*G), (Dtype)0, deriv_mu2_sums, streamId);

		} else if (this->use_gmm_square_gauss_normalization) {
			// deriv_mu2_sums = 2 * sum(gauss_dist * deriv_mu2);
			caffe_gpu_dot_batched((S*F*G) * (K_w*K_h), gauss_dist, deriv_mu2, deriv_mu2_sums, S*F*G, this->tmp_precomp_index_gpu, false, streamId);

			CUDNN_CALL_WITH_STREAM(streamId,
					caffe_gpu_scal((S*F*G), (Dtype)2, deriv_mu2_sums)
			);
		} else {
			// deriv_mu2_sums = sum(deriv_mu2);
			caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_mu2, deriv_mu2_sums, S*F*G, this->tmp_precomp_index_gpu, streamId);
		}

		// gauss_mu2_sum = abs(gauss_mu2_sum) > 1e-10 ? gauss_mu2_sum : 0;
		caffe_gpu_clip_eps(this->deriv_mu2_sums_buffer_.count(), SUM_MIN_BOUND, deriv_mu2_sums, deriv_mu2_sums, streamId);

		// use caffe_gpu_mul_batched instead of caffe_gpu_mul to use manually defined cuda stream
		caffe_gpu_mul_batched(S*F*G, gauss_norm_with_w, deriv_mu2_sums, deriv_mu2_sums,0,streamId); // deriv_mu1_sums = deriv_mu1_sums * guass_norm;


		// and add normalizaion (i.e., deriv_mu1 = deriv_mu1 - deriv_weight * deriv_mu1_sums)
		threadsPerBlock = dim3(K_w* K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
		numBlocks = dim3(1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

		axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock,0,streamId>>>(K_w * K_h, (Dtype)-1, deriv_mu2_sums, deriv_weight,  gauss_norm_with_w, deriv_mu2, S*F*G);
	}

	this->dirty_mu2_deriv_buffer = false;

	//return this->deriv_mu2_buffer_;
	return output_buffer;
}


template <typename Dtype>
__global__ void conv_gauss_sigma_deriv_kernel(const int N, const int k_w, int k_h,
												const Dtype* MU1, const Dtype* MU2, const Dtype* SIGMA_3_INV, const Dtype* guass_dist,
												Dtype* guass_deriv_sigma) {
	const int filter_size = k_w * k_h;

	for (int n = blockIdx.z * blockDim.z + threadIdx.z; n < N; n += blockDim.z * gridDim.z){

		// blockDim by x and y should always be 1 since whole filter will always fit into one block, so just retrive filter x,y indeces and calculate gaussians
		const int x = threadIdx.x;
		const int y = threadIdx.y;

		const int ptr_offset =  n * filter_size + y * k_w + x;

		// read w, mu1, mu2, sigma and other data needed to compute gaussian Distributions
		const Dtype mu1 = MU1[n];
		const Dtype mu2 = MU2[n];
		const Dtype sigma_cube_inv = SIGMA_3_INV[n];

		const Dtype gauss_value = guass_dist[ptr_offset];

		const Dtype dist_x = x - mu1;
		const Dtype dist_x_2 = dist_x*dist_x;

		const Dtype dist_y = y - mu2;
		const Dtype dist_y_2 = dist_y*dist_y;

		const Dtype dist = dist_x_2 + dist_y_2;

		guass_deriv_sigma[ptr_offset] = (dist * sigma_cube_inv) * gauss_value;
	}
}
template <typename Dtype>
shared_ptr<Blob<Dtype> > CuDNNGaussianConvLayer<Dtype>::get_sigma_derivative_filters(shared_ptr<Blob<Dtype> > output_buffer, shared_ptr<Blob<Dtype> > deriv_weight_buffer, cudaStream_t streamId) {
	// input buffers:
	//  - this->deriv_weight_buffer_ (return)			[S * G * F * K_h * K_w]
	//  - this->guass_dist_buffer_ (getter)				[S * G * F * K_h * K_w]
	//  - this->guass_norm_with_w_buffer_ (getter)		[S * G * F]
	//  - this->param_buffer_sigma_ 					[S * G * F]
	//  - this->param_buffer_sigma_cube_inv_			[S * G * F]

	// temp buffers:
	//  - this->deriv_sigma_sums_buffer_				[S * G * F] (shared)

	// output buffers:
	//  - this->deriv_sigma_buffer_ (return)			[S * G * F * K_h * K_w] (shared)

	if (this->dirty_sigma_deriv_buffer) {
		// get needed buffers
		const Dtype* gauss_dist = this->get_gauss_distribution_buffer()->gpu_data();
		const Dtype* gauss_norm_with_w = this->get_gauss_normalization_with_weight_buffer()->gpu_data();
		//const Dtype* deriv_weight = this->get_weight_derivative_filters(this->deriv_weight_buffer_)->gpu_data();
		const Dtype* deriv_weight = deriv_weight_buffer->gpu_data();

		const Dtype* gauss_params_sigma_cube_inv = this->tmp_buf.sigma_cube_inv->gpu_data(); // NOTE: this requires call to get_gauss_distribution_buffer() beforehand
		const Dtype* gauss_params_mu1 = this->param_buffer_mu1_->gpu_data();
		const Dtype* gauss_params_mu2 = this->param_buffer_mu2_->gpu_data();

		Dtype* deriv_sigma_sums = this->tmp_buf.deriv_sigma_sums->mutable_gpu_data();

		Dtype* deriv_sigma = output_buffer->mutable_gpu_data();

		const int S = this->conv_in_channels_;
		const int F = this->conv_out_channels_;
		const int G = this->NUM_GAUSS;

		const int K_w = this->kernel_w_;
		const int K_h = this->kernel_h_;

		// compute weights deriv from retrieved buffers
		dim3 threadsPerBlock = dim3(K_w, K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
		dim3 numBlocks = dim3(1, 1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

		conv_gauss_sigma_deriv_kernel<Dtype><<<numBlocks,threadsPerBlock,0,streamId>>>(S*G*F, K_w, K_h, gauss_params_mu1, gauss_params_mu2, gauss_params_sigma_cube_inv, gauss_dist, deriv_sigma);

		// compute sum of derivative with
		if (this->use_gmm_gauss_normalization == false) {
			// if there is no normalization then there should be no derivative of normalization
			caffe_gpu_set_async((S*F*G), (Dtype)0, deriv_sigma_sums, streamId);

		} else if (this->use_gmm_square_gauss_normalization) {
			// deriv_sigma_sums = 2 * sum(gauss_dist * deriv_sigma);
			caffe_gpu_dot_batched((S*F*G) * (K_w*K_h), gauss_dist, deriv_sigma, deriv_sigma_sums, S*F*G, this->tmp_precomp_index_gpu, false, streamId);

			CUDNN_CALL_WITH_STREAM(streamId,
					caffe_gpu_scal((S*F*G), (Dtype)2, deriv_sigma_sums)
			);
		} else {
			// deriv_sigma_sums = sum(deriv_sigma);
			caffe_gpu_sum((S*F*G) * (K_w*K_h), deriv_sigma, deriv_sigma_sums, S*F*G, this->tmp_precomp_index_gpu, streamId);
		}

		// use caffe_gpu_mul_batched instead of caffe_gpu_mul to use manually defined cuda stream
		caffe_gpu_mul_batched(S*F*G, gauss_norm_with_w, deriv_sigma_sums, deriv_sigma_sums,0,streamId); // deriv_mu1_sums = deriv_mu1_sums * guass_norm;

		// and add normalizaion (i.e., deriv_mu1 = deriv_mu1 - deriv_weight * deriv_mu1_sums)
		threadsPerBlock = dim3(K_w* K_h, CAFFE_CUDA_NUM_THREADS/(K_w * K_h));
		numBlocks = dim3(1, (S*F*G + threadsPerBlock.y - 1) / threadsPerBlock.y);

		axpby_kernel_batched<Dtype><<<numBlocks,threadsPerBlock,0,streamId>>>(K_w * K_h, (Dtype)-1, deriv_sigma_sums, deriv_weight,  gauss_norm_with_w, deriv_sigma, S*F*G);
	}

	this->dirty_sigma_deriv_buffer = false;

	//return this->deriv_sigma_buffer_;
	return output_buffer;
}

#define LOAD_VECTOR(OUT, IN, INDEX, N) \
	for (int i = 0; i < N; ++i) \
		OUT[i] = IN[INDEX + i];

#define SAVE_VECTOR(OUT, IN, INDEX, N) \
	for (int i = 0; i < N; ++i) \
		OUT[INDEX + i] = IN[i];

template <typename Dtype, int NUM_F_PER_THREAD, int G>
__global__ void merge_components_kernel(int S, int F, Dtype* params_w, Dtype* params_mu1, Dtype* params_mu2, Dtype* params_sigma, const Dtype threshold, const Dtype* random_direction_mu1, const Dtype* random_direction_mu2, Dtype new_sigma, Dtype* scores) {

	// threadIdx.x .. over F/NUM_F_PER_THREAD
	// threadIdx.y .. over S

	Dtype* score_dist = scores +  S*G*F * 0;
	Dtype* score_hell = scores +  S*G*F * 1;
	Dtype* score_KL = scores +  S*G*F * 2;

	for (int s = blockIdx.y * blockDim.y + threadIdx.y; s < S; s += blockDim.y * gridDim.y){
		for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < F/NUM_F_PER_THREAD; f += blockDim.x * gridDim.x){

			// TODO: repeat untill there are no overlaps - max 5 times !
			bool merging_needed = false;

			for (int g1 = 0; g1 < G; ++g1) {

				int index_g1 = (s*G +g1)*F +f*NUM_F_PER_THREAD;

				float w_g1[NUM_F_PER_THREAD],mu1_g1[NUM_F_PER_THREAD],mu2_g1[NUM_F_PER_THREAD],sigma_g1[NUM_F_PER_THREAD];

				// for g1 load from four different features (i.e. F)
				LOAD_VECTOR(w_g1, params_w, index_g1, NUM_F_PER_THREAD);
				LOAD_VECTOR(mu1_g1, params_mu1, index_g1, NUM_F_PER_THREAD);
				LOAD_VECTOR(mu2_g1, params_mu2, index_g1, NUM_F_PER_THREAD);
				LOAD_VECTOR(sigma_g1, params_sigma, index_g1, NUM_F_PER_THREAD);

				for (int g2 = g1+1; g2 < G; ++g2) {

					int index_g2 = (s*G +g2)*F +f*NUM_F_PER_THREAD;

					// for g2 load from four different features (i.e. F)
					Dtype w_g2[NUM_F_PER_THREAD],mu1_g2[NUM_F_PER_THREAD],mu2_g2[NUM_F_PER_THREAD],sigma_g2[NUM_F_PER_THREAD];

					LOAD_VECTOR(w_g2, params_w, index_g2, NUM_F_PER_THREAD);
					LOAD_VECTOR(mu1_g2, params_mu1, index_g2, NUM_F_PER_THREAD);
					LOAD_VECTOR(mu2_g2, params_mu2, index_g2, NUM_F_PER_THREAD);
					LOAD_VECTOR(sigma_g2, params_sigma, index_g2, NUM_F_PER_THREAD);

					// also load directions for new values - load them even if not needed since it would take the same time
					Dtype new_mu1[NUM_F_PER_THREAD],new_mu2[NUM_F_PER_THREAD];

					LOAD_VECTOR(new_mu1, random_direction_mu1, index_g2, NUM_F_PER_THREAD);
					LOAD_VECTOR(new_mu2, random_direction_mu2, index_g2, NUM_F_PER_THREAD);

					// compute distance

					// shouldMergeByDist
					//dist = [sum((A(2:3) - B(2:3)).^2), abs(A(4) - B(4))];
					//if B(1) == 0 || sign(B(1)) ~= sign(A(1)),
					//  b = 0;
					//elseif any((dist - thresholds) > 0)
					//	b = 0;
					//end
					Dtype dist[NUM_F_PER_THREAD];
					Dtype dist_sigma[NUM_F_PER_THREAD];

					for (int i = 0; i < NUM_F_PER_THREAD; ++i) {
						Dtype d1 = mu1_g1[i] - mu1_g2[i];
						Dtype d2 = mu2_g1[i] - mu2_g2[i];
						dist[i] = d1*d1 + d2*d2;
						dist_sigma[i] = abs(sigma_g1[i] - sigma_g2[i]);
					}

					// shouldMergeByHellinger
					//dist = 1-sqrt(2*sigma_a*sigma_b/(sigma_a.^2+sigma_b.^2)).*exp(-1/4.*(norm(mu_a-mu_b).^2/(sigma_a.^2  + sigma_b.^2)));

					//if B(1) == 0 || sign(B(1)) ~= sign(A(1)),
					//	b = 0;
					//elseif any((dist - thresholds) > 0)
					//	b = 0;
					//end

					Dtype dist_hellinger[NUM_F_PER_THREAD];

					for (int i = 0; i < NUM_F_PER_THREAD; ++i) {
						Dtype sigma_norm = 1/(sigma_g1[i]*sigma_g1[i] + sigma_g2[i]*sigma_g2[i] );
						dist_hellinger[i] = 1 - sqrt(2 * sigma_g1[i] * sigma_g2[i] * sigma_norm) * exp(-1/4.0f * dist[i] * sigma_norm  );
					}

					// shouldMergeByKullbackLeibler
					//dist = log(sigma_b/sigma_a) + (sigma_a.^2 + norm(mu_a-mu_b).^2)./(2*sigma_b.^2) - 1/2;
					//if B(1) == 0 || sign(B(1)) ~= sign(A(1)),
					//	b = 0;
					//elseif any((dist - thresholds) > 0)
					//	b = 0;
					//end
					Dtype dist_kullback_leibler[NUM_F_PER_THREAD];

					for (int i = 0; i < NUM_F_PER_THREAD; ++i) {
						dist_kullback_leibler[i] = log(sigma_g2[i]/sigma_g1[i]) + (sigma_g1[i]*sigma_g1[i] + dist[i])/(2*sigma_g2[i]*sigma_g2[i]) - 1/2.0f;
					}

					// merge two components needed
					for (int i = 0; i < NUM_F_PER_THREAD; ++i) {

						if (scores) {
							score_dist[(s*G +g1)*F +f*NUM_F_PER_THREAD + i] = dist[i];
							score_dist[(s*G +g2)*F +f*NUM_F_PER_THREAD + i] = dist[i];

							score_hell[(s*G +g1)*F +f*NUM_F_PER_THREAD + i] = dist_hellinger[i];
							score_hell[(s*G +g2)*F +f*NUM_F_PER_THREAD + i] = dist_hellinger[i];

							score_KL[(s*G +g1)*F +f*NUM_F_PER_THREAD + i] = dist_kullback_leibler[i];
							score_KL[(s*G +g2)*F +f*NUM_F_PER_THREAD + i] = dist_kullback_leibler[i];
						}
						bool should_merge = signbit(w_g1[i]) == signbit(w_g2[i]) && dist_hellinger[i] < threshold;

						if (should_merge) {
							merging_needed = true;

							// let g1 be new component and move g1 randomly away

							// combine weights by adding them
							Dtype merged_weight = w_g1[i] + w_g2[i];

							// combine mu and sigma by weighted sum
							Dtype merged_mu1 = 1./merged_weight * (mu1_g1[i] * w_g1[i] + mu1_g2[i] * w_g2[i]);
							Dtype merged_mu2 = 1./merged_weight * (mu2_g1[i] * w_g1[i] + mu2_g2[i] * w_g2[i]);
							Dtype merged_sigma = 1./merged_weight * (sigma_g1[i] * w_g1[i] + sigma_g2[i] * w_g2[i]);

							// set values for g2
							// set value to 10% of mean between weights of g1 and g2 and swicth the sign
							w_g2[i] = -1 * (w_g1[i] + w_g2[i])/2 * 0.1;

							sigma_g2[i] = new_sigma;

							// make sure component is moved at least 5-sigma away from new position (i.e., new components should have minimal contact)
							// in direction of new_mu vector
							float s = 5*(merged_sigma*merged_sigma + sigma_g2[i]*sigma_g2[i])/sqrt(new_mu1[i]*new_mu1[i] + new_mu2[i]*new_mu2[i]);
							mu1_g2[i] += s * new_mu1[i];
							mu2_g2[i] += s * new_mu2[i];


							// finally set merged value for retained component
							w_g1[i] = merged_weight;
							mu1_g1[i] = merged_mu1;
							mu2_g1[i] = merged_mu2;
							sigma_g1[i] = merged_sigma;
						}
					}

					// write out new values
					SAVE_VECTOR(params_w, w_g2, index_g2, NUM_F_PER_THREAD);
					SAVE_VECTOR(params_mu1, mu1_g2, index_g2, NUM_F_PER_THREAD);
					SAVE_VECTOR(params_mu2, mu2_g2, index_g2, NUM_F_PER_THREAD);
					SAVE_VECTOR(params_sigma, sigma_g2, index_g2, NUM_F_PER_THREAD);
				}

				// write out new values
				SAVE_VECTOR(params_w, w_g1, index_g1, NUM_F_PER_THREAD);
				SAVE_VECTOR(params_mu1, mu1_g1, index_g1, NUM_F_PER_THREAD);
				SAVE_VECTOR(params_mu2, mu2_g1, index_g1, NUM_F_PER_THREAD);
				SAVE_VECTOR(params_sigma, sigma_g1, index_g1, NUM_F_PER_THREAD);
			}
		}
	}
}


#include "caffe/filler.hpp"

template <typename Dtype>
void CuDNNGaussianConvLayer<Dtype>::fill_random_mean(Blob<Dtype>* random_mu1, Blob<Dtype>* random_mu2) {

	FillerParameter filler_param;
	filler_param.set_max(1);
	filler_param.set_min(-1);

	caffe::UniformFiller<Dtype> mean_filler(filler_param);

	if (random_mu1 != NULL) {
		mean_filler.Fill(random_mu1);

		// also set gradients to zero, since this is learnable parameter and its gradients will be added to values
		caffe_gpu_set(random_mu1->count(), (Dtype)0, random_mu1->mutable_gpu_diff());
	}
	if (random_mu2 != NULL) {
		mean_filler.Fill(random_mu2);

		// also set gradients to zero, since this is learnable parameter and its gradients will be added to values
		caffe_gpu_set(random_mu2->count(), (Dtype)0, random_mu2->mutable_gpu_diff());
	}


}

template <typename Dtype>
void CuDNNGaussianConvLayer<Dtype>::merge_components(Blob<Dtype>* scores_blob) {

	LOG(INFO) << "Doing merging optimization with threshold: " << this->gmm_merge_threshold;

#define NUM_F_PER_THREAD 4

	const int S = this->conv_in_channels_;
	const int F = this->conv_out_channels_;
	const int G = this->NUM_GAUSS;

	CHECK_EQ(0,F % NUM_F_PER_THREAD) << "Number of feature not multiple of 4 - merge_components requires a multiple of 4 number of features!";

	Dtype* params_w = this->param_buffer_w_->mutable_gpu_data();
	Dtype* params_mu1 = this->param_buffer_mu1_->mutable_gpu_data();
	Dtype* params_mu2 = this->param_buffer_mu2_->mutable_gpu_data();
	Dtype* params_sigma = this->param_buffer_sigma_->mutable_gpu_data();

	// random values should be filled beforehand - this must be done to ensure syncing bewteen multiple GPUs (buffers need to be part of learnable parameters to be sycned!)

	Dtype* random_direction_mu1 = this->tmp_buf.random_mu1->mutable_gpu_data();
	Dtype* random_direction_mu2 = this->tmp_buf.random_mu2->mutable_gpu_data();

	// read thrshold from .protobuf definition
	Dtype threshold = this->gmm_merge_threshold;

	Dtype new_sigma = this->layer_param_.convolution_param().sigma_filler().value();


	size_t dimx = std::min<size_t>(F/NUM_F_PER_THREAD,CAFFE_CUDA_NUM_THREADS);

	dim3 threadsPerBlock = dim3(dimx,
							std::min<size_t>(CAFFE_CUDA_NUM_THREADS/dimx,S) );

	dim3 numBlocks = dim3((F/NUM_F_PER_THREAD + threadsPerBlock.x - 1) / threadsPerBlock.x,
						(S + threadsPerBlock.y - 1) / threadsPerBlock.y);


	Dtype* scores = scores_blob != NULL ? scores_blob->mutable_gpu_data() : NULL;

	if (G == 2)
		merge_components_kernel<Dtype,NUM_F_PER_THREAD,2><<<numBlocks,threadsPerBlock,0,0>>>(S, F, params_w, params_mu1, params_mu2, params_sigma, threshold, random_direction_mu1, random_direction_mu2, new_sigma, scores);
	else if (G == 4)
		merge_components_kernel<Dtype,NUM_F_PER_THREAD,4><<<numBlocks,threadsPerBlock,0,0>>>(S, F, params_w, params_mu1, params_mu2, params_sigma, threshold, random_direction_mu1, random_direction_mu2, new_sigma, scores);
	else if (G == 6)
		merge_components_kernel<Dtype,NUM_F_PER_THREAD,6><<<numBlocks,threadsPerBlock,0,0>>>(S, F, params_w, params_mu1, params_mu2, params_sigma, threshold, random_direction_mu1, random_direction_mu2, new_sigma, scores);
	else if (G == 8)
		merge_components_kernel<Dtype,NUM_F_PER_THREAD,8><<<numBlocks,threadsPerBlock,0,0>>>(S, F, params_w, params_mu1, params_mu2, params_sigma, threshold, random_direction_mu1, random_direction_mu2, new_sigma, scores);
	else if (G == 16)
		merge_components_kernel<Dtype,NUM_F_PER_THREAD,16><<<numBlocks,threadsPerBlock,0,0>>>(S, F, params_w, params_mu1, params_mu2, params_sigma, threshold, random_direction_mu1, random_direction_mu2, new_sigma, scores);
	else
		CHECK_EQ(0,1) << "Unsupported number of gaussian components for merge_components; allowed only: 2, 4, 6, 8, 16";

	// prepare random values for next iterations - they will be synced among multiple GPUs (buffers need to be part of learnable parameters to be sycned!)
	fill_random_mean(this->tmp_buf.random_mu1.get(), this->tmp_buf.random_mu2.get());

	// make sure to set all buffers to dirty so they will be re-computed next time they are accessed
	set_buffers_dirty();
}

template <typename Dtype>
void CuDNNOldGaussianConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	CuDNNGaussianConvLayer<Dtype>::Forward_gpu(bottom, top);
}


template <typename Dtype>
void CuDNNOldGaussianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  bool do_mean_optmization = this->gmm_mean_iteration_step > 0 && this->current_iteration_index % this->gmm_mean_iteration_step == 0 ? true : false;
  bool do_sigma_optmization = this->gmm_sigma_iteration_step > 0 && this->current_iteration_index % this->gmm_sigma_iteration_step == 0 ? true : false;

  this->current_iteration_index++;

  const Dtype* weight = NULL;
  const Dtype* deriv_weight_kernel = NULL;
  const Dtype* deriv_mu1_kernel = NULL;
  const Dtype* deriv_mu2_kernel = NULL;
  const Dtype* deriv_sigma_kernel = NULL;

  Dtype* param_w_diff = NULL;
  Dtype* param_mu1_diff = NULL;
  Dtype* param_mu2_diff = NULL;
  Dtype* param_sigma_diff = NULL;

  if (this->param_propagate_down_[0]) {
	// weight is of size [F x S x K_h x K_w]
	weight = this->weight_buffer_->gpu_data();

	// deriv_<XYZ> are of size [S x G x F x K_h x K_w]
    deriv_weight_kernel = this->deriv_weight_buffer_->gpu_data();
    deriv_mu1_kernel = this->deriv_mu1_buffer_->gpu_data();
    deriv_mu2_kernel = this->deriv_mu2_buffer_->gpu_data();
    deriv_sigma_kernel = this->deriv_sigma_buffer_->gpu_data();

    // param_<XYZ> are of size [1 x S x G x F]
    param_w_diff = this->param_buffer_w_->mutable_gpu_diff();
    param_mu1_diff = this->param_buffer_mu1_->mutable_gpu_diff();
    param_mu2_diff = this->param_buffer_mu2_->mutable_gpu_diff();
    param_sigma_diff = this->param_buffer_sigma_->mutable_gpu_diff();
  }

  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->param_buffer_bias_->mutable_gpu_diff();
  }

  //const int I = this->num_;
  const int S = this->conv_in_channels_;
  //const int F = this->conv_out_channels_;
  //const int G = this->NUM_GAUSS;

  //const int K_w = this->kernel_w_;
  //const int K_h = this->kernel_h_;

  clock_t start_t = clock();
  for (int i = 0; i < top.size(); ++i) {
	const int top_map_size = top[i]->width() * top[i]->height();
	const int top_count = top[i]->count();

    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();

    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    Dtype* intermediate_buff = this->tmp_buffer_.mutable_gpu_data();
    Dtype* intermediate_sum_buff = this->tmp_blob_.mutable_gpu_data();

    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(this->handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
			  this->top_descs_[i],  top_diff + this->top_offset_ * g,
              cudnn::dataType<Dtype>::one,
			  this->bias_desc_, bias_diff + this->bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
    	  // use cudnnConvolutionForward to calculate gradient of weight, mean and sigma

    	  // for each sub-feature s (and for each gauss g - for now)
    	  for (int s = 0; s < S; ++s) {

    		  // do compute_parameter_deriv for d_w, d_mu1, d_mu2, d_sigma
    		  int bottom_channel_offset = bottom[i]->offset(0,s);

    		  this->compute_parameter_deriv(i, g, s, bottom_data + bottom_channel_offset, top_diff, top_count,
    				  	  	  	  	  	  	this->deriv_weight_buffer_.get(), deriv_weight_kernel,
											this->param_buffer_w_.get(), param_w_diff,
											intermediate_buff, intermediate_sum_buff, this->tmp_index_gpu, this->tmp_buffer_1_gpu, 0);


    		  if (do_mean_optmization) {
    			  this->compute_parameter_deriv(i, g, s, bottom_data + bottom_channel_offset, top_diff, top_count,
    					  	  	  	  	  	  	this->deriv_mu1_buffer_.get(), deriv_mu1_kernel,
    		  									this->param_buffer_mu1_.get(), param_mu1_diff,
    		  									intermediate_buff, intermediate_sum_buff, this->tmp_index_gpu, this->tmp_buffer_1_gpu, 4);
    			  this->compute_parameter_deriv(i, g, s, bottom_data + bottom_channel_offset, top_diff, top_count,
												this->deriv_mu2_buffer_.get(), deriv_mu2_kernel,
												this->param_buffer_mu2_.get(), param_mu2_diff,
												intermediate_buff, intermediate_sum_buff, this->tmp_index_gpu, this->tmp_buffer_1_gpu, 8);
    		  }

    		  if (do_sigma_optmization) {
				  this->compute_parameter_deriv(i, g, s, bottom_data + bottom_channel_offset, top_diff, top_count,
												this->deriv_sigma_buffer_.get(), deriv_sigma_kernel,
												this->param_buffer_sigma_.get(), param_sigma_diff,
												intermediate_buff, intermediate_sum_buff, this->tmp_index_gpu, this->tmp_buffer_1_gpu, 12);
			  }
    	  }
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        CUDNN_CHECK(cudnnConvolutionBackwardData(
        		this->handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
			  this->filter_desc_, weight + this->weight_offset_ * g,
			  this->top_descs_[i], top_diff + this->top_offset_ * g,
			  this->conv_descs_[i],
			  this->bwd_data_algo_[i], this->workspace[2*this->group_ + g],
			  this->workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
			  this->bottom_descs_[i], bottom_diff + this->bottom_offset_ * g));
      }
    }

    cudaDeviceSynchronize();
    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_gauss_conv_groups<<<1, 1>>>();
    clock_t end_t = clock();
    LOG(INFO) << "old-cudnn backward pass in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);
  }
}



template <typename Dtype>
void CuDNNOldGaussianConvLayer<Dtype>::compute_parameter_deriv(const int sample_index,
															const int group_index,
															const int subfeature_index,
															const Dtype* bottom_data,
															const Dtype* top_diff, const int top_count,
															const Blob<Dtype>* deriv_kernel, const Dtype* deriv_kernel_data,
															Blob<Dtype>* param_buffer, Dtype* param_buffer_diff,
															Dtype* intermediate_buff, Dtype* intermediate_sum_buff, int* intermediate_sum_index, int * top_remapping_index, int stream_offset) {
  const int I = this->num_;
  //const int S = this->conv_in_channels_;
  const int F = this->conv_out_channels_;
  const int G = this->NUM_GAUSS;


#define DOT_PRODUCT_AS_CUB_SUM_AND_ITERATOR 1
#define BATCHED_GAUSS 1

#ifdef BATCHED_GAUSS

  const int kernel_channel_offset = deriv_kernel->offset(subfeature_index, 0);
  const int param_channel_offset = param_buffer->offset(0, subfeature_index, 0);

  // 1. convolve [I x 1 x H x W] sub-feature inputs with [1 x G*F x K_h x K_w] deriv kernels to get [I x G*F x H x W] outputs
  CUDNN_CHECK(cudnnConvolutionForward(this->handle_[group_index], // handle
			  cudnn::dataType<Dtype>::one, // scale factor for input
			  this->backward_bottom_desc_[sample_index], bottom_data + this->bottom_offset_ * group_index, // input data; descriptor + data ptr
			  this->backward_filter_desc_, deriv_kernel_data + kernel_channel_offset + this->weight_offset_ * group_index, // filter data; descriptor + data ptr
			  this->conv_descs_[sample_index], // convolution descriptor
			  this->bwd_filter_algo_[sample_index], // algorithm selection
			  this->workspace[group_index], this->workspace_bwd_filter_sizes_[sample_index], // pre-allocated workspace
			  cudnn::dataType<Dtype>::zero, // scale factor for output
			  this->backward_intermed_desc_[sample_index], intermediate_buff + this->top_offset_ * group_index)); // output

  caffe_gpu_set(I*F*G, (Dtype)0, intermediate_sum_buff);

  // 2. multiply [I x G*F x H x W] outputs with [I x F x H x W] back-propagated error
  // 3. sum [I x G*F x H x W] into [ 1 x G*F x 1 x 1] gradients and copy to [1 x S x G x F]
#ifdef DOT_PRODUCT_AS_CUB_SUM_AND_ITERATOR
  // when DOT_PRODUCT_AS_CUB_SUM_AND_ITERATOR is defined then use implementation of dot product with
  // CUB library using segmented SUM and wrapper around input iterators to perform multiplication and rearanging of top data
  // 2+3
  caffe_gpu_dot_batched_mapped(top_count*G, top_diff, top_remapping_index,  intermediate_buff, intermediate_sum_buff, I * F*G, intermediate_sum_index);
#else
  // we can do multiplcation and sum seperately - but for higher number of features it will be slightly slower than using caffe_gpu_dot_batched_mapped
  // 2.
  caffe_gpu_mul_split(top_count * G, intermediate_buff, top_diff, intermediate_buff, top_count, top_count/I, G*top_count/I);
  // 3.
  caffe_gpu_sum(top_count*G, intermediate_buff, intermediate_sum_buff, I * F*G, intermediate_sum_index);
#endif
  caffe_gpu_sum_elementwise(I * F*G, intermediate_sum_buff, param_buffer_diff + param_channel_offset, F*G);

#else

  for (int gg = 0; gg < G; ++gg) {
	  const int kernel_channel_offset = deriv_kernel->offset(subfeature_index, gg);
	  const int param_channel_offset = param_buffer->offset(0, subfeature_index, gg);

	  // 1. convolve [I x 1 x H x W] sub-feature inputs with [1 x (G?)*F x K_h x K_w] deriv kernels to get [I x F x H x W] outputs
	  CUDNN_CHECK(cudnnConvolutionForward(handle_[group_index], // handle
				  cudnn::dataType<Dtype>::one, // scale factor for input
				  backward_bottom_desc_[sample_index], bottom_data + bottom_offset_ * group_index, // input data; descriptor + data ptr
				  backward_filter_desc_, deriv_kernel_data + kernel_channel_offset + this->weight_offset_ * group_index, // filter data; descriptor + data ptr
				  conv_descs_[sample_index], // convolution descriptor
				  bwd_filter_algo_[sample_index], // algorithm selection
				  workspace[group_index], workspace_bwd_filter_sizes_[sample_index], // pre-allocated workspace
				  cudnn::dataType<Dtype>::zero, // scale factor for output
				  backward_intermed_desc_[sample_index], intermediate_buff + top_offset_ * group_index)); // output

	  caffe_gpu_set(I*F, (Dtype)0, intermediate_sum_buff);

	  // 2. multiply [I x F x H x W] outputs with [I x F  x H x W] back-propagated error
	  // 3. sum [I x F x H x W] into [ 1 x F x 1 x 1] gradients and copy to [1 x S x G x F]
#ifdef DOT_PRODUCT_AS_CUB_SUM_AND_ITERATOR
	  // 2+3
	  caffe_gpu_dot_batched(top_count, intermediate_buff, top_diff, intermediate_sum_buff, I * F, intermediate_sum_index);
#else
	  // 2.
	  caffe_gpu_mul(top_count, top_diff, intermediate_buff, intermediate_buff);
	  // 3.
	  caffe_gpu_sum(top_count, intermediate_buff, intermediate_sum_buff, I * F, intermediate_sum_index);
#endif
	  caffe_gpu_sum_elementwise(I * F, intermediate_sum_buff, param_buffer_diff + param_channel_offset, F);

  }

#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNGaussianConvLayer);
INSTANTIATE_LAYER_GPU_FUNCS(CuDNNOldGaussianConvLayer);

}  // namespace caffe
#endif
