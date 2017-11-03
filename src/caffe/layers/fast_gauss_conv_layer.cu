#include <vector>
#include <memory>
#include <caffe/layers/gauss_conv_layer.hpp>

#include "caffe/layers/gauss_conv_layer.hpp"

#include "caffe/util/math_functions_extra.hpp"
#include "caffe/util/convolve.hpp"

#include "caffe/layers/fast_gauss/fast_gauss_forward.hpp"
#include "caffe/layers/fast_gauss/fast_gauss_backward.hpp"


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

    // before we get params we need to ensure params are within valid bounds
    {
        // we still need to ensure our values are within valid bounds
        // clip sigma, mu1 and mu2 to within bounds
        caffe_gpu_clip_lower(this->param_buffer_sigma_->count(), this->gmm_sigma_lower_bound, this->param_buffer_sigma_->mutable_gpu_data(), this->param_buffer_sigma_->mutable_gpu_data());

        caffe_gpu_clip_lower(this->param_buffer_mu1_->count(), (Dtype)this->gmm_component_border_bound, this->param_buffer_mu1_->mutable_gpu_data(), this->param_buffer_mu1_->mutable_gpu_data());
        caffe_gpu_clip_lower(this->param_buffer_mu1_->count(), (Dtype)this->gmm_component_border_bound, this->param_buffer_mu2_->mutable_gpu_data(), this->param_buffer_mu2_->mutable_gpu_data());

        caffe_gpu_clip_upper(this->param_buffer_mu1_->count(), this->kernel_w_-1 - (Dtype)this->gmm_component_border_bound, this->param_buffer_mu1_->mutable_gpu_data(), this->param_buffer_mu1_->mutable_gpu_data());
        caffe_gpu_clip_upper(this->param_buffer_mu1_->count(), this->kernel_h_-1 - (Dtype)this->gmm_component_border_bound, this->param_buffer_mu2_->mutable_gpu_data(), this->param_buffer_mu2_->mutable_gpu_data());
    }

	const int height_out = top[0]->shape(this->channel_axis_ + 1);
	const int width_out = top[0]->shape(this->channel_axis_ + 2);

	// get filter for gaussian blur step
    Blob<Dtype>* gauss_kernel_blob = this->get_gaussian_kernel(stream_[0]);
	const Dtype* gauss_kernel = gauss_kernel_blob->gpu_data();

    // get buffers for all parameters that we learn
	const Dtype* filter_weights = this->param_buffer_w_->gpu_data();
	const Dtype* filter_offsets_float_mu1 = this->param_buffer_mu1_->gpu_data();
	const Dtype* filter_offsets_float_mu2 = this->param_buffer_mu2_->gpu_data();

    cudaEvent_t memset_top, memset_filter;
    CUDA_CHECK(cudaEventCreate(&memset_top));
    CUDA_CHECK(cudaEventCreate(&memset_filter));

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

		if (gmm_use_cudnn_in_fast_aproximation_ == false) {
            caffe_gpu_set_async<Dtype>(this->num_output_* this->num_* this->height_out_* this->width_out_, (Dtype)0, top_data, paralel_streams[0]);
            caffe_gpu_set_async<Dtype>(buffer_fwd_.filtered_images_sizes_ / sizeof(float), (Dtype)0, buffer_fwd_.filtered_images, paralel_streams[1]);

            CUDA_CHECK(cudaEventRecord(memset_top, paralel_streams[0]));
            CUDA_CHECK(cudaEventRecord(memset_filter, paralel_streams[1]));

            conv2_data_desc sig_desc(1, this->channels_* this->num_, this->height_, this->width_,
									 this->channels_* this->num_*this->height_*this->width_, this->height_*this->width_, this->width_, 1);
			conv2_data_desc filt_desc(1,1,this->prefilter_h_,this->prefilter_w_,
									  this->prefilter_h_ * this->prefilter_w_, this->prefilter_h_ * this->prefilter_w_, this->prefilter_w_, 1);

			conv2_data_desc out_desc = sig_desc;

			caffe_gpu_convolve2(interm_data, out_desc,
								bottom_data, sig_desc,
								gauss_kernel, filt_desc, stream_[0]);

            CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_top, 0));
            CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_filter, 0));
		} else {
#ifdef USE_CUDNN
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
#else
            printf("Requested CuDNN in FastAproxGaussianConvLayer, nut not compiled with CuDNN support !!");
			throw std::exception();
#endif
            // if using CuDNN then use default stream to zero buffer since that buffer is used by cudnnConvolutionForward and
            // we need to wait for it to finish
            caffe_gpu_set<Dtype>(this->num_output_* this->num_* this->height_out_* this->width_out_, (Dtype)0, top_data);
            caffe_gpu_set<Dtype>(buffer_fwd_.filtered_images_sizes_ / sizeof(float), (Dtype)0, buffer_fwd_.filtered_images);
		}


        this->forward_obj->forward_pass(interm_data,
										filter_offsets_float_mu1, filter_offsets_float_mu2, filter_weights, FastGaussForward<Dtype>::SGF, this->kernel_w_, this->kernel_h_,
										top_data,
										buffer_fwd_.filtered_images,
										NULL,
										NULL,
										buffer_fwd_.filter_offsets_and_weights, stream_[0]);


		// add bias if needed
		if (this->bias_term_) {
			const Dtype* bias_data = this->param_buffer_bias_->gpu_data();
			CUDNN_CHECK(cudnnAddTensor(handle_[0],
									   cudnn::dataType<Dtype>::one,
									   bias_desc_, bias_data ,
									   cudnn::dataType<Dtype>::one,
									   top_bias_descs_[i], top_data));
		}
		// Synchronize the work across groups, each of which went into its own
		// stream, by launching an empty kernel into the default (null) stream.
		// NOLINT_NEXT_LINE(whitespace/operators)
		//sync_fast_gauss_conv_groups<<<1, 1>>>();
	}

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

	cudaEvent_t memset_top, memset_filter, memset_error;
	CUDA_CHECK(cudaEventCreate(&memset_top));
	CUDA_CHECK(cudaEventCreate(&memset_filter));
	CUDA_CHECK(cudaEventCreate(&memset_error));

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
			// TODO: if it is faster we should add zeroing to input prepare functions !!


			if (gmm_use_cudnn_in_fast_aproximation_ == false) {
                caffe_gpu_set_async(this->buffer_bwd_.filtered_images_sizes_/sizeof(Dtype), (Dtype)0, this->buffer_bwd_.filtered_images, paralel_streams[0]);
                caffe_gpu_set_async(this->buffer_bwd_.error_image_sizes_/sizeof(Dtype), (Dtype)0, this->buffer_bwd_.error_images, paralel_streams[1]);

                CUDA_CHECK(cudaEventRecord(memset_filter, paralel_streams[0]));
                CUDA_CHECK(cudaEventRecord(memset_error, paralel_streams[1]));

                conv2_data_desc sig_desc(this->channels_* this->num_, 1, this->height_, this->width_,
										 this->height_*this->width_,  this->height_*this->width_, this->width_, 1);

				conv2_data_desc filt_desc(1,this->NUM_K,this->prefilter_h_,this->prefilter_w_,
										  this->NUM_K * this->prefilter_h_ * this->prefilter_w_, this->prefilter_h_ * this->prefilter_w_, this->prefilter_w_, 1);

				conv2_data_desc out_desc(this->channels_* this->num_, this->NUM_K, this->height_, this->width_,
										 this->height_*this->width_ * this->NUM_K,  this->height_*this->width_, this->width_, 1);

				caffe_gpu_convolve2(interm_data, out_desc,
									bottom_data, sig_desc,
									deriv_kernels_data, filt_desc, stream_[0]);


                CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_filter, 0));
                CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_error, 0));

			} else {
#ifdef USE_CUDNN
				// perform pre-filtering for each parameter i.e. with four different derivative filters
				CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
													cudnn::dataType<Dtype>::one,
													bottom_descs_[i], bottom_data,
													bwd_prefilter_kernel_desc_, deriv_kernels_data,
													bwd_conv_data_descs_[i],
													bwd_data_algo_[i], workspace[0], workspace_bwd_data_sizes_[i],
													cudnn::dataType<Dtype>::zero,
													bwd_interm_data_descs_[i], interm_data));

#else
				printf("Requested CuDNN in FastAproxGaussianConvLayer, nut not compiled with CuDNN support !!");
				throw std::exception();
#endif
                // if using CuDNN then use default stream to zero buffer since that buffer is used by cudnnConvolutionForward and
                // we need to wait for it to finish
                caffe_gpu_set(this->buffer_bwd_.filtered_images_sizes_/sizeof(Dtype), (Dtype)0, this->buffer_bwd_.filtered_images);
                caffe_gpu_set(this->buffer_bwd_.error_image_sizes_/sizeof(Dtype), (Dtype)0, this->buffer_bwd_.error_images);

            }

			// collect gradients by shifting convolved bottom input data and multiplying it with the top error data

            // WARNING: if this->kernel_w_ or this->kernel_h_ changes then memory will not be allocated properly
			backward_grad_obj->backward_pass(interm_data, top_error,
									   filter_offsets_float_mu1, filter_offsets_float_mu2,
									   filter_weights, this->kernel_w_, this->kernel_h_,
									   bwd_gradients_data,
									   this->buffer_bwd_.filtered_images,
									   this->buffer_bwd_.error_images,
									   this->buffer_bwd_.filter_weights,
									   this->buffer_bwd_.filter_offsets,
									   //this->ignore_edge_gradients_, stream_[0]);
                                       this->ignore_edge_gradients_, 0);

		}


		// finally perform back-propagation of the error values
		if (propagate_down[i]) {

            Dtype const* top_error_for_bwd = top_error;

            // if size top_error (input) is smaller then interm_data (output)  (i.e. expected input should be the same size as output)
			// then we need to copy top_error to bigger buffer i.e. with padded zeros
			if (buffer_bwd_.resized_top_for_bwd_sizes_ > 0) {
				// set zeros
				caffe_gpu_set_async<Dtype>(buffer_bwd_.resized_top_for_bwd_sizes_ / sizeof(float), (Dtype)0, buffer_bwd_.resized_top_for_bwd, stream_[0]);

				// then copy but with appropriate padding
				caffe_gpu_pad2d(this->num_ * this->num_output_, this->height_out_, this->width_out_, this->width_/2 - this->width_out_/2, top_error, buffer_bwd_.resized_top_for_bwd, stream_[0]);

                top_error_for_bwd = buffer_bwd_.resized_top_for_bwd;
			}
			if (gmm_use_cudnn_in_fast_aproximation_ == false) {

                // NOTE: memory buffer is shared with gradient compute so make sure not to zero it before backward_grad_obj->backward_pass is done

                caffe_gpu_set_async<Dtype>(this->channels_* this->num_* this->height_* this->width_, (Dtype)0, bottom_error, paralel_streams[0]);
                caffe_gpu_set_async<Dtype>(buffer_fwd_.filtered_images_sizes_ / sizeof(float), (Dtype)0, buffer_fwd_.filtered_images, paralel_streams[1]);

                CUDA_CHECK(cudaEventRecord(memset_top, paralel_streams[0]));
                CUDA_CHECK(cudaEventRecord(memset_filter, paralel_streams[1]));

				int max_width = std::max(this->width_out_,this->width_);
				int max_height = std::max(this->height_out_,this->height_);

				conv2_data_desc sig_desc(1, this->num_output_* this->num_, max_height, max_width,
										 this->num_output_* this->num_*max_height*max_width, max_height*max_width, max_width, 1);

				conv2_data_desc filt_desc(1,1,this->prefilter_h_,this->prefilter_w_,
										  this->prefilter_h_ * this->prefilter_w_, this->prefilter_h_ * this->prefilter_w_, this->prefilter_w_, 1);

				conv2_data_desc out_desc = sig_desc;


				caffe_gpu_convolve2(interm_data, out_desc,
                                    top_error_for_bwd, sig_desc,
									deriv_error_kernel, filt_desc, stream_[0]);

                CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_top, 0));
                CUDA_CHECK(cudaStreamWaitEvent(stream_[0], memset_filter, 0));

			} else {
#ifdef USE_CUDNN
				// we need to do pre-filtering of the error values
				CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
													cudnn::dataType<Dtype>::one,
													top_descs_[i], top_error_for_bwd,
													fwd_prefilter_kernel_desc_, deriv_error_kernel,
													bwd_conv_error_descs_[i],
													bwd_error_algo_[i], workspace[0], workspace_bwd_error_sizes_[i],
													cudnn::dataType<Dtype>::zero,
													bwd_interm_error_descs_[i], interm_data));
#else
				printf("Requested CuDNN in FastAproxGaussianConvLayer, nut not compiled with CuDNN support !!");
			throw std::exception();
#endif
                // if using CuDNN then use default stream to zero buffer since that buffer is used by cudnnConvolutionForward and
                // we need to wait for it to finish
                caffe_gpu_set<Dtype>(this->channels_* this->num_* this->height_* this->width_, (Dtype)0, bottom_error);
                caffe_gpu_set<Dtype>(buffer_fwd_.filtered_images_sizes_ / sizeof(float), (Dtype)0, buffer_fwd_.filtered_images);
			}
			// then use our custom kernel for forwarding, however we need to transpose kernels, which in our case means
			// that we need to rotate mu1,mu2 locations


			// get param buffer for mu1 and mu2 that will be rotated
			Dtype *param_mu1_backprop = this->tmp_param_buffer_.mutable_gpu_data() + 0 * param_size;
			Dtype *param_mu2_backprop = this->tmp_param_buffer_.mutable_gpu_data() + 1 * param_size;

			// rot(mu) = (kernel_w-1) - mu
			{
				caffe_gpu_memcpy_async(param_size * sizeof(float), filter_offsets_float_mu1, param_mu1_backprop, 0);
				caffe_gpu_memcpy_async(param_size * sizeof(float), filter_offsets_float_mu2, param_mu2_backprop, 0);

				caffe_gpu_scal(param_size, (Dtype)-1, param_mu1_backprop);
				caffe_gpu_scal(param_size, (Dtype)-1, param_mu2_backprop);

				caffe_gpu_add_scalar(param_size, (Dtype)(this->kernel_w_ - 1), param_mu1_backprop);
				caffe_gpu_add_scalar(param_size, (Dtype)(this->kernel_h_ - 1), param_mu2_backprop);
			}


			// now we take the blured error data and perform sum over shifted input data with our custom kernel i.e. forward pass
			this->backward_backporp_obj->forward_pass(interm_data,
													  param_mu1_backprop, param_mu2_backprop, filter_weights, FastGaussForward<Dtype>::FGS, this->kernel_w_, this->kernel_h_,
													  bottom_error,
													  buffer_fwd_.filtered_images,
													  NULL,
													  NULL,
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
