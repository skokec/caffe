#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/gauss_conv_layer.hpp"
#include "caffe/util/math_functions_extra.hpp"

namespace caffe {

__global__ void sync_gauss_conv_groups() { }

template <typename Dtype>
void CuDNNGaussianConvLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* weight = this->weight_buffer_->gpu_data();
  //cudaDeviceSynchronize();

  clock_t start_t = clock();

  // compile guassian parameters into kernels for regular CNN
  //if (this->current_iteration_index == 0)
  this->precompute_guassian_weights_gpu(true);
  //return ;
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
    //	clock_t start_t = clock();
      // Filters.
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

  //cudaDeviceSynchronize();
  clock_t end_t = clock();
}

template <typename Dtype>
void CuDNNGaussianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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

  caffe_gpu_set(this->param_buffer_w_->count(), (Dtype)0, param_w_diff);
  caffe_gpu_set(this->param_buffer_mu1_->count(), (Dtype)0, param_mu1_diff);
  caffe_gpu_set(this->param_buffer_mu2_->count(), (Dtype)0, param_mu2_diff);
  caffe_gpu_set(this->param_buffer_sigma_->count(), (Dtype)0, param_sigma_diff);

  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->param_buffer_bias_->mutable_gpu_diff();
  }

  const int I = this->num_;
  const int S = this->conv_in_channels_;
  const int F = this->conv_out_channels_;
  const int G = this->NUM_GAUSS;

  const int K_w = this->kernel_w_;
  const int K_h = this->kernel_h_;

  clock_t start_t = clock();
  for (int i = 0; i < top.size(); ++i) {
	const int top_map_size = top[i]->width() * top[i]->height();
	const int top_count = top[i]->count();

    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();

    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
    	  // use cudnnConvolutionForward to calculate gradient of weight, mean and sigma

		  filterActs_YxX_color(bottom_data, top_diff, this->deriv_weight_buffer_.get()->gpu_data(), param_w_diff,
										  I, S, F, G, 0, 0, 0,
										  this->conv_in_width_, this->conv_in_height_,
										  this->width_out_, this->height_out_,
										  this->kernel_w_, this->kernel_h_,
										  this->pad_w_, 0, paralel_streams[0]);

		  if (do_mean_optmization) {
			  filterActs_YxX_color(bottom_data, top_diff, this->deriv_mu1_buffer_.get()->gpu_data(), param_mu1_diff,
														  I, S, F, G, 0, 0, 0,
														  this->conv_in_width_, this->conv_in_height_,
														  this->width_out_, this->height_out_,
														  this->kernel_w_, this->kernel_h_,
														  this->pad_w_, 0, paralel_streams[1]);

			  filterActs_YxX_color(bottom_data, top_diff, this->deriv_mu2_buffer_.get()->gpu_data(), param_mu2_diff,
														  I, S, F, G, 0, 0, 0,
														  this->conv_in_width_, this->conv_in_height_,
														  this->width_out_, this->height_out_,
														  this->kernel_w_, this->kernel_h_,
														  this->pad_w_, 0, paralel_streams[2]);
		  }

		  if (do_sigma_optmization) {
			  filterActs_YxX_color(bottom_data, top_diff, this->deriv_sigma_buffer_.get()->gpu_data(), param_sigma_diff,
														  I, S, F, G, 0, 0, 0,
														  this->conv_in_width_, this->conv_in_height_,
														  this->width_out_, this->height_out_,
														  this->kernel_w_, this->kernel_h_,
														  this->pad_w_, 0, paralel_streams[3]);
		  }
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    //cudaDeviceSynchronize();
    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_gauss_conv_groups<<<1, 1>>>();
    clock_t end_t = clock();
    //LOG(INFO) << "old-cudnn backward pass in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);
  }
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
