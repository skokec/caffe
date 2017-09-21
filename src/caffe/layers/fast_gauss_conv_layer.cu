#ifdef USE_CUDNN
#include <vector>
#include <memory>

#include "caffe/layers/gauss_conv_layer.hpp"

#include "caffe/util/math_functions_extra.hpp"
#include "caffe/util/custom_cub.cuh"

#include "caffe/util/fast_gauss_forward.hpp"
#include "caffe/util/fast_gauss_backward.hpp"

namespace caffe {

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	// - first perform gaussian bluring based on variance that is fixed over the whole layer (use CuDNN for that)
	// - then perform forward pass with our custom kernel
	// - optionally add bias

	clock_t start_t = clock();

	// check if we need to do merging of components;
	// make sure we check based on steps done in backpropagation and we should avoid merging if only forward is called (by default current_iteration_index=0 so start at second iter
	bool do_merginig_optmization = this->gmm_merge_iteration_step > 0 && (this->current_iteration_index + 1) % this->gmm_merge_iteration_step == 0 ? true : false;

	// if during training then merge components if needed
	if (do_merginig_optmization) {
		//merge_components();
	}

	const int height_out = top[0]->shape(this->channel_axis_ + 1);
	const int width_out = top[0]->shape(this->channel_axis_ + 2);

	// get filter for gaussian blur step
	const Dtype* gauss_kernel = this->get_gaussian_kernel(stream_[0])->gpu_data();

	// get buffers for all parameters that we learn
	const Dtype* filter_weights = this->param_buffer_w_->gpu_data();
	const Dtype* filter_offsets_float_mu1 = this->param_buffer_mu1_->gpu_data();
	const Dtype* filter_offsets_float_mu2 = this->param_buffer_mu2_->gpu_data();

	/*
	plot_blob_data(*this->param_buffer_w_);
	plot_blob_data(*this->param_buffer_mu1_);
	plot_blob_data(*this->param_buffer_mu2_);
	*/

	for (int i = 0; i < bottom.size(); ++i) {
		// input data
		const Dtype* bottom_data = bottom[i]->gpu_data();

		// intermediate data for blurred input
		Dtype* interm_data = interm_buffer_.mutable_gpu_data();

		// actual output data
		Dtype* top_data = top[i]->mutable_gpu_data();


		// Forward through cuDNN and our custom kernel
		/*
		cudaDeviceSynchronize();
		clock_t start_t = clock();
		*/

		// first perform convolutions with gaussian filter (i.e. gaussian blur)
		// we use cudnn forward implementation by casting
		//  - input into [N*S x 1 x HxW]
		//  - weight into [1 x K_h x K_w]
		// this effectively perform single convolution on each input feature and we get
		//  - output [N*S x 1 x HxW] => [N x S x HxW]
		CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
											cudnn::dataType<Dtype>::one,
											bottom_descs_[i], bottom_data,
											fwd_prefilter_kernel_desc_, gauss_kernel,
											fwd_conv_descs_[i],
											fwd_algo_[i], workspace[0], workspace_fwd_sizes_[i],
											cudnn::dataType<Dtype>::zero,
											fwd_interm_descs_[i], interm_data));
		//interm_data = (Dtype*)bottom_data;
		/*
        //cudaMemset(buffer_fwd_.filter_offsets_and_weights, 0, buffer_fwd_.filter_weights_sizes_ + buffer_fwd_.filter_offsets_sizes_);
		cudaDeviceSynchronize();
		clock_t end_t = clock();
		std::cout << "gaussian pre-filtering in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
		// now we take the blured input data and perform sum over shifted input data with our custom kernel
		*/
        /*start_t = clock();*/
		caffe::fast_gauss_forward<Dtype>(interm_data,
										 filter_offsets_float_mu1, filter_offsets_float_mu2, filter_weights, FAST_GAUSS_PARAM_SGF,
										 top_data,
										 this->num_, this->channels_, this->num_output_, this->NUM_GAUSS,
										 this->width_out_, this->height_out_,
										 this->kernel_w_, this->kernel_h_,
										 this->use_interpolation_,
										 buffer_fwd_.filtered_images,0,
										 NULL,0,
										 NULL,0,
										 buffer_fwd_.filter_offsets_and_weights, stream_[0]);
        /*
        cudaDeviceSynchronize();
        end_t = clock();
        std::cout << "fast_gauss_forward in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
         */

		// add bias if needed
		if (this->bias_term_) {
			const Dtype* bias_data = this->param_buffer_bias_->gpu_data();
			CUDNN_CHECK(cudnnAddTensor(handle_[0],
									   cudnn::dataType<Dtype>::one,
									   bias_desc_, bias_data ,
									   cudnn::dataType<Dtype>::one,
									   top_bias_descs_[i], top_data));
		}
		cudaDeviceSynchronize();
		// Synchronize the work across groups, each of which went into its own
		// stream, by launching an empty kernel into the default (null) stream.
		// NOLINT_NEXT_LINE(whitespace/operators)
		//sync_fast_gauss_conv_groups<<<1, 1>>>();
	}
	//cudaDeviceSynchronize();


}


template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
													 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	//  - first convolve bottom input data with kernels for individual parameters (w, mu1, mu2, sigma)
	//  - then compute and collect gradients by shifting convolved bottom input data and multiplying it with the top error data
	//  - finally back-propagade the error by convolving top error with the rotated filters (we can use the same function as for forward-pass, but need to transpose mu1 and mu2 values)

    this->current_iteration_index++;

	// get buffers for all parameters that we learn
	const Dtype* filter_weights = this->param_buffer_w_->gpu_data();
	const Dtype* filter_offsets_float_mu1 = this->param_buffer_mu1_->gpu_data();
	const Dtype* filter_offsets_float_mu2 = this->param_buffer_mu2_->gpu_data();

	Dtype* param_weights_diff = this->param_buffer_w_->mutable_gpu_diff();
	Dtype* param_mu1_diff = this->param_buffer_mu1_->mutable_gpu_diff();
	Dtype* param_mu2_diff = this->param_buffer_mu2_->mutable_gpu_diff();
	Dtype* param_sigma_diff = this->param_buffer_sigma_->mutable_gpu_diff();

	Dtype* bias_diff = this->param_buffer_bias_->mutable_gpu_diff();

	Dtype* bwd_gradients_data = this->bwd_gradients.mutable_gpu_data();

	// get filter for gaussian blur step
	const Dtype* deriv_w_kernel = this->get_deriv_kernel_weight(stream_[0])->gpu_data();
	const Dtype* deriv_mu1_kernel = this->get_deriv_kernel_mu1(stream_[0])->gpu_data();
	const Dtype* deriv_mu2_kernel = this->get_deriv_kernel_mu2(stream_[0])->gpu_data();
	const Dtype* deriv_sigma_kernel = this->get_deriv_kernel_sigma(stream_[0])->gpu_data();

	const Dtype* deriv_error_kernel = this->get_deriv_kernel_error(stream_[0])->gpu_data();

	// copy all four kernels into a single blob
	Dtype* deriv_kernels_data  = prefilter_deriv_kernels_.mutable_gpu_data();

	const int prefilter_size = this->prefilter_h_ * this->prefilter_w_ ;

	if (NUM_K > 0) caffe_gpu_memcpy(prefilter_size * sizeof(float), deriv_w_kernel, deriv_kernels_data + 0 * prefilter_size);
	if (NUM_K > 1) caffe_gpu_memcpy(prefilter_size * sizeof(float), deriv_mu1_kernel, deriv_kernels_data + 1 * prefilter_size);
	if (NUM_K > 2) caffe_gpu_memcpy(prefilter_size * sizeof(float), deriv_mu2_kernel, deriv_kernels_data + 2 * prefilter_size);
	if (NUM_K > 3) caffe_gpu_memcpy(prefilter_size * sizeof(float), deriv_sigma_kernel, deriv_kernels_data + 3 * prefilter_size);

	// intermediate data for blurred input
	Dtype* interm_data = interm_buffer_.mutable_gpu_data();

	// transform all four accumulated gradients into seperabe buffers of size [S x G x F]
	int param_size = this->NUM_GAUSS * this->conv_in_channels_ * this->conv_out_channels_;

	// make sure gradient accumulation buffer is zeroed
	caffe_gpu_memset(param_size * NUM_K * sizeof(Dtype), 0, bwd_gradients_data);

	for (int i = 0; i < bottom.size(); ++i) {

		// input data
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* bottom_error = bottom[i]->mutable_gpu_diff();

		// actual output data
		const Dtype* top_data = top[i]->gpu_data();
		const Dtype* top_error = top[i]->gpu_diff();

		// Gradient w.r.t. bias.
		if (this->bias_term_ && this->param_propagate_down_[1]) {
			CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0],
													 cudnn::dataType<Dtype>::one,
													 top_bias_descs_[i],  top_error,
													 cudnn::dataType<Dtype>::one,
													 bias_desc_, bias_diff));

		}

		// Gradient w.r.t w,mu1,mu2 and sigma
		if (this->param_propagate_down_[0]) {
			// perform pre-filtering for each parameter i.e. with four different derivative filters
			CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
												cudnn::dataType<Dtype>::one,
												bottom_descs_[i], bottom_data,
												bwd_prefilter_kernel_desc_, deriv_kernels_data,
												bwd_conv_data_descs_[i],
												bwd_data_algo_[i], workspace[0], workspace_bwd_data_sizes_[i],
												cudnn::dataType<Dtype>::zero,
												bwd_interm_data_descs_[i], interm_data));

			// TODO: if it is faster we should add zeroing to input prepare functions !!
			CUDA_CHECK(cudaMemsetAsync(this->buffer_bwd_.filtered_images,0,this->buffer_bwd_.filtered_images_sizes_, stream_[0]));
			CUDA_CHECK(cudaMemsetAsync(this->buffer_bwd_.error_images,0,this->buffer_bwd_.error_image_sizes_, stream_[0]));
			/*cudaDeviceSynchronize();

			// TODO: update support for K=4 as well
            clock_t start_t = clock();*/
			// collect gradients by shifting convolved bottom input data and multiplying it with the top error data
			caffe::fast_gauss_backward_multi_subfeatures<Dtype>(interm_data, top_error,
																filter_offsets_float_mu1, filter_offsets_float_mu2,
																filter_weights, bwd_gradients_data,
																this->num_, this->channels_, this->num_output_, this->NUM_GAUSS, NUM_K,
																this->width_out_, this->height_out_,
																this->kernel_w_, this->kernel_h_,
																this->use_interpolation_, this->ignore_edge_gradients_,
																this->buffer_bwd_.filtered_images,0,
																this->buffer_bwd_.error_images,0,
																this->buffer_bwd_.filter_weights,0,
																this->buffer_bwd_.filter_offsets,0, stream_[0]);
			/*cudaDeviceSynchronize();
			clock_t end_t = clock();
            std::cout << "fast_gauss_backward_multi_subfeatures in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;*/

		}


		// finally perform back-propagation of the error values
		if (propagate_down[i]) {

			// we need to do pre-filtering of the error values
			CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
												cudnn::dataType<Dtype>::one,
												top_descs_[i], top_error,
												fwd_prefilter_kernel_desc_, deriv_error_kernel,
												bwd_conv_error_descs_[i],
												bwd_error_algo_[i], workspace[0], workspace_bwd_error_sizes_[i],
												cudnn::dataType<Dtype>::zero,
												bwd_interm_error_descs_[i], interm_data));

			// then use our custom kernel for forwarding, however we need to transpose kernels, which in our case means
			// that we need to rotate mu1,mu2 locations

			// get param buffer for mu1 and mu2 that will be rotated
			Dtype *param_mu1_backprop = this->tmp_param_buffer_.mutable_gpu_data() + 0 * param_size;
			Dtype *param_mu2_backprop = this->tmp_param_buffer_.mutable_gpu_data() + 1 * param_size;

			// rot(mu) = (kernel_w-1) - mu
			{
				caffe_gpu_memcpy(param_size * sizeof(float), filter_offsets_float_mu1, param_mu1_backprop);
				caffe_gpu_memcpy(param_size * sizeof(float), filter_offsets_float_mu2, param_mu2_backprop);

				caffe_gpu_scal(param_size, (Dtype)-1, param_mu1_backprop);
				caffe_gpu_scal(param_size, (Dtype)-1, param_mu2_backprop);

				caffe_gpu_add_scalar(param_size, (Dtype)(this->kernel_w_ - 1), param_mu1_backprop);
				caffe_gpu_add_scalar(param_size, (Dtype)(this->kernel_h_ - 1), param_mu2_backprop);
			}

			// now we take the blured error data and perform sum over shifted input data with our custom kernel i.e. forward pass
			caffe::fast_gauss_forward<Dtype>(interm_data,
											 param_mu1_backprop, param_mu2_backprop, filter_weights, FAST_GAUSS_PARAM_FGS,
                                             bottom_error,
											 this->num_, this->num_output_, this->channels_, this->NUM_GAUSS,
											 this->width_out_, this->height_out_,
											 this->kernel_w_, this->kernel_h_,
											 this->use_interpolation_,
											 buffer_fwd_.filtered_images,0,
											 NULL,0,
											 NULL,0,
											 buffer_fwd_.filter_offsets_and_weights, stream_[0]);

		}
	}
	// we need to accumulate gradients to the final buffer and add weights to some derivates
	if (this->param_propagate_down_[0]) {
		// multiply gradients with appropriate weights
		/// add add weight multiplyer as specifed by derivative formula only for mu1,mu2 and sigma
		if (NUM_K > 1) caffe_gpu_mul(param_size, bwd_gradients_data + 1 * param_size, filter_weights, bwd_gradients_data + 1 * param_size); // mu1
		if (NUM_K > 2) caffe_gpu_mul(param_size, bwd_gradients_data + 2 * param_size, filter_weights, bwd_gradients_data + 2 * param_size); // mu2
		if (NUM_K > 3) caffe_gpu_mul(param_size, bwd_gradients_data + 3 * param_size, filter_weights, bwd_gradients_data + 3 * param_size); // sigma

		// for weight gradient we only accumulate to final buffer
		if (NUM_K > 0) caffe_gpu_axpy(param_size, (Dtype)1, bwd_gradients_data + 0 * param_size, param_weights_diff); // w
		if (NUM_K > 1) caffe_gpu_axpy(param_size, (Dtype)1, bwd_gradients_data + 1 * param_size, param_mu1_diff); // mu1
		if (NUM_K > 2) caffe_gpu_axpy(param_size, (Dtype)1, bwd_gradients_data + 2 * param_size, param_mu2_diff); // mu2
		if (NUM_K > 3) caffe_gpu_axpy(param_size, (Dtype)1, bwd_gradients_data + 3 * param_size, param_sigma_diff); // sigma
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(FastAproxGaussianConvLayer);


}  // namespace caffe
#endif
