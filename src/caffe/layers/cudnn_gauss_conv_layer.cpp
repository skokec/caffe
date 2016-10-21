#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

//#include "caffe/vision_layers.hpp"
#include "caffe/layers/gauss_conv_layer.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define GAUSS_CUDNN_STREAMS_PER_GROUP 2
// for legacy !!
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNGaussianConvLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  BaseGaussianConvLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(1, this->group_)
        << "CuDNNGaussianConvLayer does not support group parameter at the moment";

  // Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[this->group_ * GAUSS_CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * GAUSS_CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * GAUSS_CUDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * GAUSS_CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);

  top_descs_.clear();
  bottom_descs_.clear();
  conv_descs_.clear();

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  this->num_guass_per_compute = this->NUM_GAUSS;

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;

  paralel_streams = new cudaStream_t[4];
  for (int g = 0; g < 4; ++g) {
	  cudaStreamCreate(&paralel_streams[g]);
  }
}

template <typename Dtype>
void CuDNNGaussianConvLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  if (this->bottom_dim_ == bottom[0]->count(this->channel_axis_)  && top[0]->count() > 0 && this->top_dim_ == top[0]->count(this->channel_axis_) &&
		  this->num_ == bottom[0]->num() &&
		  this->height_ == bottom[0]->height() &&
		  this->width_ == bottom[0]->width() ) {
	  return;
  }

  BaseGaussianConvLayer<Dtype>::Reshape(bottom, top);

  // we should remove all buffers from BaseGaussianConvLayer and replace them with our own buffers
  // however Blob has lazy allocation so no memory will be allocated if it is not used (also important not to remove them since they can be used in unit tests)
  /*
  {
	this->param_buffer_sigma_square_inv_.reset();
	this->param_buffer_sigma_cube_inv_.reset();
	this->param_buffer_sigma_square_inv_half_.reset();
	this->weight_buffer_.reset();
	this->deriv_error_buffer_.reset();
	this->deriv_weight_buffer_.reset();
	this->deriv_sigma_buffer_.reset();
	this->deriv_mu1_buffer_.reset();
	this->deriv_mu2_buffer_.reset();
	this->is_weight_enabled_buffer_.reset();
	this->tmp_deriv_weight_buffer_.reset();
	this->guass_dist_buffer_.reset();
	this->gauss_dist_square_buffer_.reset();
	this->deriv_mu1_times_gauss_dist_buffer_.reset();
	this->deriv_mu2_times_gauss_dist_buffer_.reset();
	this->deriv_sigma_times_gauss_dist_buffer_.reset();

	this->guass_norm_buffer_.reset();
	this->deriv_mu1_sums_buffer_.reset();
	this->deriv_mu2_sums_buffer_.reset();
	this->deriv_sigma_sums_buffer_.reset();
	this->weight_vert_buffer_.reset();
	this->weight_horiz_buffer_.reset();
  }
  */
  // [S * G * F * K_h * K_w]
  kernel_buf.weights.reset(new Blob<Dtype>(this->weight_buffer_->shape()));
  kernel_buf.deriv_weight.reset(new Blob<Dtype>(this->deriv_weight_buffer_->shape()));
  kernel_buf.deriv_mu1.reset(new Blob<Dtype>(this->deriv_mu1_buffer_->shape()));
  kernel_buf.deriv_mu2.reset(new Blob<Dtype>(this->deriv_mu2_buffer_->shape()));
  kernel_buf.deriv_sigma.reset(new Blob<Dtype>(this->deriv_sigma_buffer_->shape()));

  tmp_buf.random_mu1.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));
  tmp_buf.random_mu2.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));

  // prepare random values for next merging iteration;
  // buffers need to be synced between multiple GPUs and are therefore flaged as learnable parameters to induce copying from root GPU to child GPUs
  fill_random_mean(this->tmp_buf.random_mu1.get(), this->tmp_buf.random_mu2.get());

#ifndef NDEBUG
  // connect weight buffers to blobs when in debug mode so that they can be inspected by matlab
  int extra_blobs_offset = this->blobs_.size() - this->num_extra_blobs;

  this->blobs_[extra_blobs_offset + 0] = kernel_buf.weights;
  //this->blobs_[extra_blobs_offset + 1] = kernel_buf.deriv_error;
  this->blobs_[extra_blobs_offset + 2] = kernel_buf.deriv_weight;
  this->blobs_[extra_blobs_offset + 3] = kernel_buf.deriv_mu1;
  this->blobs_[extra_blobs_offset + 4] = kernel_buf.deriv_mu2;
  this->blobs_[extra_blobs_offset + 5] = kernel_buf.deriv_sigma;

  if (this->layer_param_.convolution_param().gmm_legacy_merge_blobs() == false) {
	  this->blobs_[extra_blobs_offset + 6] = tmp_buf.random_mu1;
  	  this->blobs_[extra_blobs_offset + 7] = tmp_buf.random_mu2;
  }
#else
  kernel_buf.weights->ReshapeLike(*kernel_buf.deriv_weight);
#endif

  tmp_buf.distribution.reset(new Blob<Dtype>(kernel_buf.deriv_weight->shape()));

  // [S * G * F]
  tmp_buf.sigma_square_inv.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));
  tmp_buf.sigma_cube_inv.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));
  tmp_buf.sigma_square_inv_half.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));

  tmp_buf.norm.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));
  tmp_buf.norm_with_w.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));

  tmp_buf.deriv_mu1_sums.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));
  tmp_buf.deriv_mu2_sums.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));
  tmp_buf.deriv_sigma_sums.reset(new Blob<Dtype>(this->param_buffer_w_->shape()));


  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNGaussianConvLayer input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";

  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;

  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);

    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // choose backward algo for data
	CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
		filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
		CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
		workspace_limit_bytes, &bwd_data_algo_[i]));

	// get workspace size
	CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
		filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
		bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
	total_workspace_fwd        = std::max(total_workspace_fwd,
									   workspace_fwd_sizes_[i]);
	total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
									   workspace_bwd_data_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
							   total_workspace_bwd_data);

  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
								 (this->group_ * GAUSS_CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    LOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * GAUSS_CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * GAUSS_CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
CuDNNGaussianConvLayer<Dtype>::~CuDNNGaussianConvLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * GAUSS_CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  for (int g = 0; g < 4; ++g) {
	  cudaStreamDestroy(paralel_streams[g]);
  }

  cudaFree(workspaceData);
  delete [] stream_;
  delete [] paralel_streams;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
}

template <typename Dtype>
void CuDNNOldGaussianConvLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CuDNNGaussianConvLayer<Dtype>::LayerSetUp(bottom, top);

  // Initialize algorithm arrays
  bwd_filter_algo_= new cudnnConvolutionFwdAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    bwd_filter_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    // default algorithms don't require workspace
    workspace_bwd_filter_sizes_[i] = 0;
  }

  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
    const int kernel_h = kernel_shape_data[0];
    const int kernel_w = kernel_shape_data[1];

  // Descriptors for backward pass:
  // Kernel descriptor should have only one chanel
  cudnn::createFilterDesc<Dtype>(&backward_filter_desc_, this->num_output_ * this->num_guass_per_compute/ this->group_, 1, kernel_h, kernel_w);

  // input descriptor and intermediate output will need to have different size then other descriptors
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t backward_bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&backward_bottom_desc);
    backward_bottom_desc_.push_back(backward_bottom_desc);
    cudnnTensorDescriptor_t backward_intermed_desc;
    cudnn::createTensor4dDesc<Dtype>(&backward_intermed_desc);
    backward_intermed_desc_.push_back(backward_intermed_desc);
  }
}

template <typename Dtype>
void CuDNNOldGaussianConvLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  if (this->bottom_dim_ == bottom[0]->count(this->channel_axis_)  && top[0]->count() > 0 && this->top_dim_ == top[0]->count(this->channel_axis_) &&
		  this->num_ == bottom[0]->num() &&
		  this->height_ == bottom[0]->height() &&
		  this->width_ == bottom[0]->width() ) {
	  return;
  }

  CuDNNGaussianConvLayer<Dtype>::Reshape(bottom, top);

  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {

    // create tensors for backward
    cudnn::setTensor4dDesc<Dtype>(&backward_bottom_desc_[i],
		this->num_, 1, height, width,
		this->channels_ * height * width, height * width, width, 1);

	cudnn::setTensor4dDesc<Dtype>(&backward_intermed_desc_[i],
		this->num_,
		this->num_output_ * this->num_guass_per_compute / this->group_, height_out, width_out,
		this->num_output_ * this->num_guass_per_compute * this->out_spatial_dim_,
		this->out_spatial_dim_, width_out, 1);

	// choose backward algorithm for filter
	CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(this->handle_[0],
			this->backward_bottom_desc_[i],
			this->backward_filter_desc_,
			this->conv_descs_[i],
			this->backward_intermed_desc_[i],
	  CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
	  workspace_limit_bytes,
	  &this->bwd_filter_algo_[i]));

	CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(this->handle_[0],
			this->backward_bottom_desc_[i],
			this->backward_filter_desc_,
			this->conv_descs_[i],
			this->backward_intermed_desc_[i],
			this->bwd_filter_algo_[i],
	  &(this->workspace_bwd_filter_sizes_[i])));
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
	total_workspace_fwd        = std::max(total_workspace_fwd,
											this->workspace_fwd_sizes_[i]);
	total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
											this->workspace_bwd_data_sizes_[i]);
	total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
											this->workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
							   total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
								 (this->group_ * CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > this->workspaceSizeInBytes) {
    LOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    this->workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), this->workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
    	  this->workspace_fwd_sizes_[i] = 0;
    	  this->workspace_bwd_filter_sizes_[i] = 0;
    	  this->workspace_bwd_data_sizes_[i] = 0;
    	  this->fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    	  this->bwd_filter_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    	  this->bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
    	  this->workspace[g] = NULL;
      }
      // NULL out underlying data
      this->workspaceData = NULL;
      this->workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
    	this->workspace[g] = reinterpret_cast<char *>(this->workspaceData) + g*max_workspace;
    }
  }


  // resize temporary buffers from parent
  this->tmp_buffer_.Reshape(this->num_, this->conv_out_channels_ * this->num_guass_per_compute, this->height_out_, this->width_out_);
  memset(&this->tmp_buffer_1, 0, sizeof(this->tmp_buffer_1)); // HACK: there is a bug in Blob; shape_data is not zeroed correctly so we do this manually
  this->tmp_buffer_1.Reshape(this->num_, this->conv_out_channels_ * this->num_guass_per_compute, this->height_out_, this->width_out_);

  int* tmp_buffer_1_cpu = this->tmp_buffer_1.mutable_cpu_data();

  int index = 0;
  for (int i = 0; i < this->num_; ++i) {
	  for (int g = 0; g < this->num_guass_per_compute; ++g) {
		  for (int j = 0; j < this->conv_out_channels_ * this->height_out_* this->width_out_; j++) {
			  tmp_buffer_1_cpu[index++] = i * (this->conv_out_channels_ * this->height_out_* this->width_out_) + j;
		  }
	  }
  }
  this->tmp_buffer_1_gpu = this->tmp_buffer_1.mutable_gpu_data();

  this->tmp_blob_.Reshape(1, 1, this->num_, this->conv_out_channels_ * this->num_guass_per_compute);

  // pre-computed offset indexes for batched sums (when using caffe_gpu_sum)
  this->tmp_index_.Reshape(1, 1, 1,  this->num_ *this->num_guass_per_compute* this->conv_out_channels_ + 1);

  int* tmp_index_cpu = this->tmp_index_.mutable_cpu_data();

  tmp_index_cpu[0] = 0;

  for (int i = 0; i < this->tmp_index_.count()-1; i++) tmp_index_cpu[i+1] = this->height_out_ * this->width_out_*(i+1);

  this->tmp_index_gpu = this->tmp_index_.mutable_gpu_data();

}

template <typename Dtype>
CuDNNOldGaussianConvLayer<Dtype>::~CuDNNOldGaussianConvLayer() {

	// Check that handles have been setup before destroying.
	if (!this->handles_setup_) { return; }

	for (int i = 0; i < backward_bottom_desc_.size(); i++) {

		cudnnDestroyTensorDescriptor(backward_bottom_desc_[i]);
		cudnnDestroyTensorDescriptor(backward_intermed_desc_[i]);
	}
	cudnnDestroyFilterDescriptor(backward_filter_desc_);

	delete [] bwd_filter_algo_;
	delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNGaussianConvLayer);
INSTANTIATE_CLASS(CuDNNOldGaussianConvLayer);

}   // namespace caffe
#endif
