#include <algorithm>
#include <vector>

#include "caffe/layers/gauss_conv_layer.hpp"

#include "caffe/util/math_functions_extra.hpp"

#include "caffe/layers/fast_gauss/fast_gauss_forward.hpp"
#include "caffe/layers/fast_gauss/fast_gauss_backward.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <caffe/layers/gauss_conv_layer.hpp>

namespace caffe {

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    // we setup with allowing num of gauss only as a factor of two (due to CUDA implementation)
    // (when computing gradinets we neet to make sure we ignore last one)
    this->allowed_gauss_div = 2;

    BaseGaussianConvLayer<Dtype>::LayerSetUp(bottom, top);

    CHECK_EQ(1, this->group_) << "CuDNNGaussianConvLayer does not support group parameter at the moment";


    // Initialize CUDA streams
    stream_         = new cudaStream_t[this->group_];

    // workspace data
    workspaceSizeInBytes = 0;
    workspaceData = NULL;
    workspace = new void*[this->group_ ];

    // initialize size arrays
    workspace_fwd_sizes_ = new size_t[bottom.size()];
    workspace_bwd_data_sizes_ = new size_t[bottom.size()];
    workspace_bwd_error_sizes_ = new size_t[bottom.size()];

    for (int g = 0; g < this->group_ ; g++) {
        workspace[g] = NULL;
    }

    for (size_t i = 0; i < bottom.size(); ++i) {
        // default algorithms don't require workspace
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
    }

    buffer_fwd_.filtered_images_sizes_ = 0;
    buffer_fwd_.filter_weights_sizes_ = 0;
    buffer_fwd_.filter_offsets_sizes_ = 0;

    buffer_bwd_.filtered_images_sizes_ = 0;
    buffer_bwd_.error_image_sizes_= 0;
    buffer_bwd_.filter_weights_sizes_ = 0;
    buffer_bwd_.filter_offsets_sizes_ = 0;

    buffer_fwd_.filter_offsets = NULL;
    buffer_fwd_.filter_weights = NULL;
    buffer_fwd_.filter_offsets_and_weights = NULL;
    buffer_fwd_.filtered_images = NULL;

    buffer_bwd_.filter_offsets = NULL;
    buffer_bwd_.filter_weights = NULL;
    buffer_bwd_.error_images = NULL;
    buffer_bwd_.filtered_images = NULL;

    // Set the indexing parameters.
    bias_offset_ = (this->num_output_ / this->group_);

    // Create filter descriptor.
    const int* kernel_shape_data = this->kernel_shape_.cpu_data();

    Dtype sigma = *this->param_buffer_sigma_->cpu_data();

    // define pre-filtering kernel size based on 5*sigma - NOTE: currently this is fixed and cannot be changed if sigma increases !!
    prefilter_h_ = 2 * ceil(5 * sigma) + 1;
    prefilter_w_ = 2 * ceil(5 * sigma) + 1;

    // we need to ensure to get the same intermediate size as input size so pad accordingly
    prefilter_pad_h_ = floor(prefilter_h_ / 2);
    prefilter_pad_w_ = floor(prefilter_w_ / 2);

    // we allow only stride of 1 and should apply stride in offset/sum (NOTE: stride is currently unsupported)
    prefilter_stride_h_ = 1;
    prefilter_stride_w_ = 1;

    // create buffers used to generate kernels (i.e. we need only one kernel
    prefilter_param_w_.Reshape(1,1,1,1);
    prefilter_param_mu1_.Reshape(1,1,1,1);
    prefilter_param_mu2_.Reshape(1,1,1,1);
    prefilter_param_sigma_.Reshape(1,1,1,1);

    // by default we generate kernels with w=1, mu=(0,0) so fill buffers with them
    // NOTE: mu=(0,0) is center of kernel so use that value
    *(prefilter_param_w_.mutable_cpu_data()) = 1.0f;
    *(prefilter_param_mu1_.mutable_cpu_data()) = prefilter_h_/2;
    *(prefilter_param_mu2_.mutable_cpu_data()) = prefilter_h_/2;

    this->use_interpolation_ = true;

    paralel_streams = new cudaStream_t[4];
    for (int g = 0; g < 4; ++g) {
        cudaStreamCreate(&paralel_streams[g]);
    }

    this->gmm_use_cudnn_in_fast_aproximation_ = this->layer_param_.convolution_param().gmm_use_cudnn_in_fast_aproximation();

#ifdef USE_CUDNN

    // Initialize CUDA streams and cuDNN.
    handle_         = new cudnnHandle_t[this->group_ ];

    // Initialize algorithm arrays
    fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
    bwd_data_algo_  = new cudnnConvolutionFwdAlgo_t[bottom.size()];
    bwd_error_algo_ = new cudnnConvolutionFwdAlgo_t[bottom.size()];


    for (size_t i = 0; i < bottom.size(); ++i) {
        // initialize all to default algorithms
        fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
        bwd_data_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
        bwd_error_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    }

    for (int g = 0; g < this->group_ ; g++) {
        CUDA_CHECK(cudaStreamCreate(&stream_[g]));
        CUDNN_CHECK(cudnnCreate(&handle_[g]));

        CUDNN_CHECK (cudnnSetStream(handle_[g], stream_[g]));
    }

    cudnn::createFilterDesc<Dtype>(&fwd_prefilter_kernel_desc_,
                                   1, 1, prefilter_h_, prefilter_w_);

    cudnn::createFilterDesc<Dtype>(&bwd_prefilter_kernel_desc_,
                                   NUM_K, 1, prefilter_h_, prefilter_w_);

    top_bias_descs_.clear();
    top_descs_.clear();
    bottom_descs_.clear();

    fwd_interm_descs_.clear();
    bwd_interm_data_descs_.clear();

    fwd_conv_descs_.clear();

    bwd_conv_data_descs_.clear();
    bwd_conv_error_descs_.clear();

    // Create tensor descriptor(s) for data and corresponding convolution(s).
    for (int i = 0; i < bottom.size(); i++) {
        {
            // descriptor for convolution during forward process
            cudnnTensorDescriptor_t bottom_desc;
            cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
            bottom_descs_.push_back(bottom_desc);

            cudnnTensorDescriptor_t output_desc;
            cudnn::createTensor4dDesc<Dtype>(&output_desc);
            fwd_interm_descs_.push_back(output_desc);

            cudnnTensorDescriptor_t top_bias_desc;
            cudnn::createTensor4dDesc<Dtype>(&top_bias_desc);
            top_bias_descs_.push_back(top_bias_desc);

            cudnnTensorDescriptor_t top_desc;
            cudnn::createTensor4dDesc<Dtype>(&top_desc);
            top_descs_.push_back(top_desc);

            cudnnConvolutionDescriptor_t conv_desc;
            cudnn::createConvolutionDesc<Dtype>(&conv_desc);
            fwd_conv_descs_.push_back(conv_desc);
        }
        {
            // descriptor for convolution during backward process
            cudnnTensorDescriptor_t interm_data_desc;
            cudnn::createTensor4dDesc<Dtype>(&interm_data_desc);
            bwd_interm_data_descs_.push_back(interm_data_desc);

            cudnnTensorDescriptor_t interm_error_desc;
            cudnn::createTensor4dDesc<Dtype>(&interm_error_desc);
            bwd_interm_error_descs_.push_back(interm_error_desc);

            cudnnConvolutionDescriptor_t conv_data_desc;
            cudnn::createConvolutionDesc<Dtype>(&conv_data_desc);
            bwd_conv_data_descs_.push_back(conv_data_desc);

            cudnnConvolutionDescriptor_t conv_error_desc;
            cudnn::createConvolutionDesc<Dtype>(&conv_error_desc);
            bwd_conv_error_descs_.push_back(conv_error_desc);
        }
    }

    // Tensor descriptor for bias.
    if (this->bias_term_) {
        cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
    }

#endif

    handles_setup_ = true;

    this->gmm_store_filter_blobs_ = this->layer_param_.convolution_param().gmm_store_filter_blobs();

}

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    if (this->bottom_dim_ == bottom[0]->count(this->channel_axis_)  && top[0]->count() > 0 && this->top_dim_ == top[0]->count(this->channel_axis_) &&
        this->num_ == bottom[0]->num() &&
        this->height_ == bottom[0]->height() &&
        this->width_ == bottom[0]->width() ) {
        return;
    }

    BaseGaussianConvLayer<Dtype>::Reshape(bottom, top);

    CHECK_EQ(1, this->stride_h_) << "CuDNNGaussianConvLayer does not support stride>1 parameter at the moment";
    CHECK_EQ(1, this->stride_w_) << "CuDNNGaussianConvLayer does not support stride>1 parameter at the moment";

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

    const int max_width = std::max(this->width_out_,this->width_);
    const int max_height = std::max(this->height_out_,this->height_);

    // Specify workspace limit for kernels directly until we have a
    // planning strategy and a rewrite of Caffe's GPU memory mangagement
    size_t workspace_limit_bytes = 8*1024*1024;

    // use inter buffer for both fwd and bwd passes so allocate buffer with suitable size for both
    interm_buffer_.Reshape(this->num_, std::max(this->channels_ * this->NUM_K, this->num_output_), max_height, max_width);

    this->prefilter_kernel_.Reshape(1,1,this->prefilter_h_, this->prefilter_w_);

    this->prefilter_deriv_kernel_weight_.ReshapeLike(this->prefilter_kernel_);
    this->prefilter_deriv_kernel_mu1_.ReshapeLike(this->prefilter_kernel_);
    this->prefilter_deriv_kernel_mu2_.ReshapeLike(this->prefilter_kernel_);
    this->prefilter_deriv_kernel_sigma_.ReshapeLike(this->prefilter_kernel_);
    this->prefilter_deriv_kernel_error_.ReshapeLike(this->prefilter_kernel_);

    this->prefilter_deriv_kernels_.Reshape(1,NUM_K,this->prefilter_h_, this->prefilter_w_);

    // we will be using precompute_guassian_weights_gpu() so setup sizes of all temporary buffers that will be used
    this->guass_dist_buffer_.ReshapeLike(this->prefilter_deriv_kernel_weight_);
    this->gauss_dist_square_buffer_.ReshapeLike(this->prefilter_deriv_kernel_weight_);
    this->deriv_mu1_times_gauss_dist_buffer_.ReshapeLike(this->prefilter_deriv_kernel_weight_);
    this->deriv_mu2_times_gauss_dist_buffer_.ReshapeLike(this->prefilter_deriv_kernel_weight_);
    this->deriv_sigma_times_gauss_dist_buffer_.ReshapeLike(this->prefilter_deriv_kernel_weight_);

    this->guass_norm_buffer_.Reshape(1, 1, 1, 1);
    this->deriv_mu1_sums_buffer_.ReshapeLike(this->guass_norm_buffer_);
    this->deriv_mu2_sums_buffer_.ReshapeLike(this->guass_norm_buffer_);
    this->deriv_sigma_sums_buffer_.ReshapeLike(this->guass_norm_buffer_);

    // pre-computed offset indexes for batched sums since we are using precompute_guassian_weights_gpu() to create filters
    // we can reuse this->tmp_precomp_index_ but need set it up with different kernel size
    this->create_precompute_index(this->tmp_precomp_index_, 1, this->prefilter_h_ * this->prefilter_w_);


    this->bwd_gradients.Reshape(this->NUM_K, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_);

    // temporary buffer used during the back-propagation of the error where we rotate mu1 and mu2
    this->tmp_param_buffer_.Reshape(2, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_);


    forward_obj.reset(new caffe::FastGaussForward<Dtype>(this->width_, this->height_, this->width_out_, this->height_out_, this->num_, this->channels_, this->num_output_, this->NUM_GAUSS, this->use_interpolation_));
    forward_obj->get_allocation_sizes(this->kernel_w_, this->kernel_h_,
                                      &buffer_fwd_.filtered_images_sizes_,
                                      &buffer_fwd_.filter_weights_sizes_,
                                      &buffer_fwd_.filter_offsets_sizes_);

    // check how much memory do we need for our custom kernels
    backward_grad_obj.reset(new caffe::FastGaussBackward<Dtype>(this->width_, this->height_, this->width_out_, this->height_out_, this->num_, this->channels_, this->num_output_, this->NUM_GAUSS, this->NUM_K, this->last_k_optional, this->use_interpolation_));

    // WARNING: if this->kernel_w_ or this->kernel_h_ changes then memory will not be allocated properly so we should use here
    //          maximum kernel_w_ and kernel_h_ allowed
    backward_grad_obj->get_allocation_sizes(this->kernel_w_, this->kernel_h_,
                                            &buffer_bwd_.filtered_images_sizes_,
                                            &buffer_bwd_.error_image_sizes_,
                                            &buffer_bwd_.filter_weights_sizes_,
                                            &buffer_bwd_.filter_offsets_sizes_);
    // for gradient accumulation

    // for error back-propagation
    // we use the same buffer as for forward pass but can be shared, just ensure buffer can accomodate both sizes
    size_t filtered_error_sizes_, filter_error_weights_sizes_, filter_error_offsets_sizes_;

    backward_backporp_obj.reset(new caffe::FastGaussForward<Dtype>(max_width, max_height, max_width, max_height, this->num_, this->num_output_, this->channels_, this->NUM_GAUSS, this->use_interpolation_));
    backward_backporp_obj->get_allocation_sizes(this->kernel_w_, this->kernel_h_,
                                                &filtered_error_sizes_,
                                                &filter_error_weights_sizes_,
                                                &filter_error_offsets_sizes_);

    buffer_bwd_.resized_top_for_bwd_sizes_ = 0;

    if (this->width_out_ != this->width_ || this->height_out_ != this->height_) {
        buffer_bwd_.resized_top_for_bwd_sizes_ = this->num_ * this->num_output_* max_height*max_width * sizeof(Dtype);
    }

    // this ensures that buffers will accomodate both fast_gauss_forward functions one used for forward pass and the second one used of error back-propagation
    buffer_fwd_.filtered_images_sizes_ = std::max(buffer_fwd_.filtered_images_sizes_, filtered_error_sizes_);
    buffer_fwd_.filter_weights_sizes_ = std::max(buffer_fwd_.filter_weights_sizes_, filter_error_weights_sizes_);
    buffer_fwd_.filter_offsets_sizes_ = std::max(buffer_fwd_.filter_offsets_sizes_, filter_error_offsets_sizes_);

    // reduce over all workspace sizes to get a maximum to allocate / reallocate
    size_t total_workspace_fwd = 0;
    size_t total_workspace_bwd_data = 0;
    size_t total_workspace_bwd_error = 0;

    total_workspace_fwd         = std::max(total_workspace_fwd,
                                           buffer_fwd_.filtered_images_sizes_ +
                                           buffer_fwd_.filter_weights_sizes_ +
                                           buffer_fwd_.filter_offsets_sizes_);
    total_workspace_bwd_data    = std::max(total_workspace_bwd_data,
                                           buffer_bwd_.filtered_images_sizes_ +
                                           buffer_bwd_.error_image_sizes_ +
                                           buffer_bwd_.filter_weights_sizes_ +
                                           buffer_bwd_.filter_offsets_sizes_ );

    total_workspace_bwd_data    = std::max(total_workspace_bwd_data,
                                           buffer_bwd_.resized_top_for_bwd_sizes_);

#ifdef USE_CUDNN
    for (int i = 0; i < bottom.size(); i++) {

        cudnn::setTensor4dDesc<Dtype>(&top_bias_descs_[i],
                                      this->num_ ,
                                      this->num_output_, height_out, width_out,
                                      this->num_output_ * height_out * width_out,
                                      height_out * width_out, width_out, 1);

        cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
                                      this->num_ *this->num_output_, 1, height_out, width_out,
                                      1 * height_out * width_out,
                                      height_out * width_out, width_out, 1);

        cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
                                      this->num_ *this->channels_, 1, height, width,
                                      1 * height * width,
                                      height * width, width, 1);

        // set descriptors for CuDNN specific for forward
        {

            cudnn::setTensor4dDesc<Dtype>(&fwd_interm_descs_[i],
                                          this->num_ *this->channels_, 1, height, width,
                                          1 * height * width,
                                          height * width, width, 1);

            cudnn::setConvolutionDesc<Dtype>(&fwd_conv_descs_[i], bottom_descs_[i],
                                             fwd_prefilter_kernel_desc_, prefilter_pad_h_, prefilter_pad_w_,
                                             prefilter_stride_h_, prefilter_stride_w_);

            // choose forward and backward algorithms + workspace(s)
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
                                                            bottom_descs_[i],
                                                            fwd_prefilter_kernel_desc_,
                                                            fwd_conv_descs_[i],
                                                            fwd_interm_descs_[i],
                                                            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                            workspace_limit_bytes,
                                                            &fwd_algo_[i]));

            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
                                                                bottom_descs_[i],
                                                                fwd_prefilter_kernel_desc_,
                                                                fwd_conv_descs_[i],
                                                                fwd_interm_descs_[i],
                                                                fwd_algo_[i],
                                                                &(workspace_fwd_sizes_[i])));
        }
        // set descriptors for CuDNN specific for backward
        {
            // descriptor and algo for pre-filetring of input data with derivative filters
            cudnn::setTensor4dDesc<Dtype>(&bwd_interm_data_descs_[i],
                                          this->num_ *this->channels_, NUM_K, height, width,
                                          NUM_K * height * width,
                                          height * width, width, 1);

            cudnn::setConvolutionDesc<Dtype>(&bwd_conv_data_descs_[i], bottom_descs_[i],
                                             bwd_prefilter_kernel_desc_, prefilter_pad_h_, prefilter_pad_w_,
                                             prefilter_stride_h_, prefilter_stride_w_);

            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
                                                            bottom_descs_[i],
                                                            bwd_prefilter_kernel_desc_,
                                                            bwd_conv_data_descs_[i],
                                                            bwd_interm_data_descs_[i],
                                                            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                            workspace_limit_bytes,
                                                            &bwd_data_algo_[i]));

            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
                                                                bottom_descs_[i],
                                                                bwd_prefilter_kernel_desc_,
                                                                bwd_conv_data_descs_[i],
                                                                bwd_interm_data_descs_[i],
                                                                bwd_data_algo_[i],
                                                                &(workspace_bwd_data_sizes_[i])));

            // descriptor and algo for pre-filetring of error data with reversed convolution filter

            cudnn::setTensor4dDesc<Dtype>(&bwd_interm_error_descs_[i],
                                          this->num_ * this->num_output_, 1, height_out, width_out,
                                          1 * height_out * width_out,
                                          height_out * width_out, width_out, 1);

            cudnn::setConvolutionDesc<Dtype>(&bwd_conv_error_descs_[i], top_descs_[i],
                                             bwd_prefilter_kernel_desc_, prefilter_pad_h_, prefilter_pad_w_,
                                             prefilter_stride_h_, prefilter_stride_w_);

            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
                                                            top_descs_[i],
                                                            fwd_prefilter_kernel_desc_,
                                                            bwd_conv_error_descs_[i],
                                                            bwd_interm_error_descs_[i],
                                                            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                            workspace_limit_bytes,
                                                            &bwd_error_algo_[i]));

            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
                                                                top_descs_[i],
                                                                fwd_prefilter_kernel_desc_,
                                                                bwd_conv_data_descs_[i],
                                                                bwd_interm_error_descs_[i],
                                                                bwd_error_algo_[i],
                                                                &(workspace_bwd_error_sizes_[i])));

        }
    }

    // Tensor descriptor for bias.
    if (this->bias_term_) {
        cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
                                      1, this->num_output_ , 1, 1);
    }


    for (size_t i = 0; i < bottom.size(); i++) {
        total_workspace_fwd        = std::max(total_workspace_fwd,
                                              workspace_fwd_sizes_[i]);
        total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                              workspace_bwd_data_sizes_[i]);
        total_workspace_bwd_error   = std::max(total_workspace_bwd_error,
                                              workspace_bwd_error_sizes_[i]);
    }
#endif

    // get max over all operations
    size_t max_workspace = std::max(total_workspace_fwd,
                                    total_workspace_bwd_data);

    max_workspace = std::max(max_workspace,
                             total_workspace_bwd_error);

    // ensure all groups have enough workspace
    size_t total_max_workspace = max_workspace * (this->group_ );

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
#ifdef USE_CUDNN
                fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
                bwd_data_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#endif
            }

            // NULL out all workspace pointers
            for (int g = 0; g < (this->group_ ); g++) {
                workspace[g] = NULL;
            }
            // NULL out underlying data
            workspaceData = NULL;
            workspaceSizeInBytes = 0;
        }

        // if we succeed in the allocation, set pointer aliases for workspaces
        for (int g = 0; g < (this->group_ ); g++) {
            workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
        }

        // NOTE: buffer_ is not ready for multiple groups so modify this if you want to have multiple groups
        // we reuse the buffer allocated for convolution, however that buffer will bi small
        // TODO:
        //    - we need to implement a mechanizem to share workspace buffer between layers
        //    - this would work for layer that are executed in sequence but will fail for parallel layers
        //    - also make sure to count on multi-GPU implementations so do explicitly use only static buffers !!!

        // We should implement memory as:
        // - for each GPU-device we use seperate class for handling memory
        // - we use static array within this class to store memory storage classes and save them with GPU-id
        // - we can get GPU-id of current active device since that is setup for us by caffe
        // - memory storage class needs to take care of:
        //   - allocating memory when requested
        //   - ensuring the same memory is used across all alocations
        //   - if more memory is requested then currently allocated then we need to allocat new memory
        //   - in case of allocating additional memory we NEED to ensure that existing instances with pointers to that memory are updated
        //     (this should be done by wraping allocated pointer around new class, possibly std:shared_ptr and using that one instead of raw pointers)
        // - we need to ensure to deallocate memory when all not used any more
        //   - when each FastAproxGaussianConvLayer is destroyed it should indicate to memory manegement class to destroy its memory
        //   - memory manegment should keep an index if valid pointer that are still using it and it should deallocate CUDA memory when there are no more valid pointers!!


        // TODO: make all memory align to 4x 32bit values
        buffer_fwd_.filtered_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData));
        buffer_fwd_.filter_offsets_and_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_);
        buffer_fwd_.filter_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_);
        buffer_fwd_.filter_offsets = reinterpret_cast<int*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_ + buffer_fwd_.filter_weights_sizes_);


        buffer_bwd_.filtered_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData));
        buffer_bwd_.error_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_);
        buffer_bwd_.filter_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_ + buffer_bwd_.error_image_sizes_);
        buffer_bwd_.filter_offsets = reinterpret_cast<int*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_ + buffer_bwd_.error_image_sizes_ + buffer_bwd_.filter_weights_sizes_);

        // we can reuse workspace data since it will not be used at the same time
        buffer_bwd_.resized_top_for_bwd = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData));
    }


    // we do not need blobs from parent version as they only take up unneccessary memory
    // but due to legacy models we retain them unless explicity requested not to
    if (this->gmm_store_filter_blobs_ == false) {
        this->weight_buffer_->Reshape(1, 1, 1, 1);
        this->deriv_error_buffer_->Reshape(1, 1, 1, 1);
        this->deriv_weight_buffer_->Reshape(1, 1, 1, 1);
        this->deriv_sigma_buffer_->Reshape(1, 1, 1, 1);
        this->deriv_mu1_buffer_->Reshape(1, 1, 1, 1);
        this->deriv_mu2_buffer_->Reshape(1, 1, 1, 1);
        this->random_mu1_buffer_->Reshape(1, 1, 1, 1);
        this->random_mu2_buffer_->Reshape(1, 1, 1, 1);
    }
}

template <typename Dtype>
FastAproxGaussianConvLayer<Dtype>::~FastAproxGaussianConvLayer() {
    // Check that handles have been setup before destroying.
    if (!handles_setup_) { return; }

    for (int g = 0; g < this->group_ ; g++) {
        cudaStreamDestroy(stream_[g]);
    }

    for (int g = 0; g < 4; ++g) {
        cudaStreamDestroy(paralel_streams[g]);
    }

    cudaFree(workspaceData);
    delete [] stream_;
    delete [] paralel_streams;
    delete [] workspace;
    delete [] workspace_fwd_sizes_;
    delete [] workspace_bwd_data_sizes_;
    delete [] workspace_bwd_error_sizes_;

#ifdef USE_CUDNN
    for (int i = 0; i < bottom_descs_.size(); i++) {
        cudnnDestroyTensorDescriptor(top_bias_descs_[i]);

        cudnnDestroyTensorDescriptor(bottom_descs_[i]);
        cudnnDestroyTensorDescriptor(top_descs_[i]);
        cudnnDestroyTensorDescriptor(fwd_interm_descs_[i]);
        cudnnDestroyConvolutionDescriptor(fwd_conv_descs_[i]);

        cudnnDestroyTensorDescriptor(bwd_interm_data_descs_[i]);
        cudnnDestroyTensorDescriptor(bwd_interm_error_descs_[i]);
        cudnnDestroyConvolutionDescriptor(bwd_conv_data_descs_[i]);
        cudnnDestroyConvolutionDescriptor(bwd_conv_error_descs_[i]);

    }
    if (this->bias_term_) {
        cudnnDestroyTensorDescriptor(bias_desc_);
    }
    cudnnDestroyFilterDescriptor(fwd_prefilter_kernel_desc_);
    cudnnDestroyFilterDescriptor(bwd_prefilter_kernel_desc_);

    for (int g = 0; g < this->group_ ; g++) {
        cudnnDestroy(handle_[g]);
    }

    delete [] handle_;
    delete [] fwd_algo_;
    delete [] bwd_data_algo_;
    delete [] bwd_error_algo_;
#endif

}

__global__ void sync_fast_gauss_conv_groups() { }

template <typename Dtype>
void plot_blob_data(Blob<Dtype>& b) {
    const Dtype* d = b.cpu_data();
    for (int n = 0;  n< b.shape(0); ++n) {
        for (int c = 0;  c< b.shape(1); ++c) {
            for (int j = 0;  j< b.shape(2); ++j) {
                for (int i = 0;  i< b.shape(3); ++i) {
                    printf("%.2f ", d[b.offset(n,c,j,i)]);
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}


template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::update_prefiltering_kernels(cudaStream_t stream) {

    // if we changed the variance then re-compute the gaussian kernel
    // we assume there is only one sigma for the whole layer !!!
    Dtype const* sigma_cpu = this->param_buffer_sigma_->cpu_data();

    if (std::fabs(current_prefilter_sigma - *sigma_cpu) > 1e-5) {

        // we compute kernels for blur using the same code as in std-implementation but we compute only for a single
        // component i.e., num_in_channels = 1, num_out_channels = 1, num_gauss = 1, and we use weight=1, mu = [0,0]

        bool is_backward_pass = true;
        bool gmm_discretize_mean = false;
        Dtype gmm_component_border_bound = 0;

        this->do_precompute_guassian_weights_gpu(this->prefilter_param_w_,
                                                 this->prefilter_param_mu1_,
                                                 this->prefilter_param_mu2_,
                                                 *this->param_buffer_sigma_.get(),
                                                 1, 1, 1,
                                                 this->prefilter_h_, this->prefilter_w_,
                                                 is_backward_pass,
                                                 this->use_gmm_weight_normalization,
                                                 this->use_gmm_square_gauss_normalization,
                                                 gmm_discretize_mean,
                                                 this->gmm_sigma_lower_bound,
                                                 gmm_component_border_bound,
                                                 this->tmp_precomp_index_,
                                                 &this->prefilter_kernel_,
                                                 NULL, NULL, NULL,
                                                 &this->prefilter_deriv_kernel_error_,
                                                 &this->prefilter_deriv_kernel_weight_,
                                                 &this->prefilter_deriv_kernel_mu1_,
                                                 &this->prefilter_deriv_kernel_mu2_,
                                                 &this->prefilter_deriv_kernel_sigma_);

        this->current_prefilter_sigma = sigma_cpu[0];


        //for debug write kernel with 1 only at center i.e. identity convolution kernel
        if (0) {
            Dtype*  gauss_kernel = this->prefilter_kernel_.mutable_cpu_data();
            Dtype*  deriv_weight_kernel = this->prefilter_deriv_kernel_weight_.mutable_cpu_data();
            Dtype*  deriv_mu1_kernel = this->prefilter_deriv_kernel_mu1_.mutable_cpu_data();
            Dtype*  deriv_mu2_kernel = this->prefilter_deriv_kernel_mu2_.mutable_cpu_data();
            Dtype*  deriv_sigma_kernel = this->prefilter_deriv_kernel_sigma_.mutable_cpu_data();
            Dtype*  deriv_error_kernel = this->prefilter_deriv_kernel_error_.mutable_cpu_data();


            int h_half = prefilter_h_/2;
            int w_half = prefilter_h_/2;
            int index = 0;
            for (int j = -h_half; j <= h_half; ++j) {
                for (int i = -w_half; i <= w_half; ++i) {

                    Dtype val = (i == 0 && j == 0 ? 1 : 0);

                    gauss_kernel[index] = val;
                    deriv_weight_kernel[index] = val;
                    deriv_mu1_kernel[index] = val;
                    deriv_mu2_kernel[index] = val;
                    deriv_sigma_kernel[index] = val;
                    deriv_error_kernel[index] = val;

                    index++;
                }
            }
        }
/*
        plot_blob_data(this->prefilter_kernel_);
        plot_blob_data(this->prefilter_deriv_kernel_weight_);
        plot_blob_data(this->prefilter_deriv_kernel_mu1_);
        plot_blob_data(this->prefilter_deriv_kernel_mu2_);
        plot_blob_data(this->prefilter_deriv_kernel_sigma_);
        plot_blob_data(this->prefilter_deriv_kernel_error_);
*/


    }
}

template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_gaussian_kernel(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->prefilter_kernel_;
}

template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_weight(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->prefilter_deriv_kernel_weight_;
}
template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_mu1(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->prefilter_deriv_kernel_mu1_;
}
template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_mu2(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->prefilter_deriv_kernel_mu2_;
}
template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_sigma(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->prefilter_deriv_kernel_sigma_;
}
template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_error(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->prefilter_deriv_kernel_error_;
}

template <typename Dtype>
void offset_and_sum_opencv(const Dtype* input_data,
                    const Dtype* filter_weights, const Dtype* filter_offsets_float_mu1, const Dtype* filter_offsets_float_mu2,
                    Dtype* output_data,
                    const int num_, const int conv_in_channels_, const int NUM_GAUSS, const int conv_out_channels_,
                    const int width_, const int height_,
                    const int width_out_, const int height_out_, const int kernel_width, const int kernel_height,
                   const FastGaussForward<float>::PARAM_FORMAT INPUT_FORMAT = FastGaussForward<float>::SGF) {

    // perform offset and sum over individual outputs
#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

    const int INTERPOlATION_Dx = 2;
    const int INTERPOlATION_Dy = 2;

    const int F_BATCH = 8;
    const int S_BATCH = 1;

    for (int n = 0; n < num_; ++n) {
        //printf("n=%d\n",n);

        cv::Mat interm_mat(conv_in_channels_ * height_,width_, CV_32F, (Dtype*)input_data + n * conv_in_channels_ * width_ * height_);
        cv::Mat top_mat(conv_out_channels_ * height_out_, width_out_, CV_32F, output_data + n * conv_out_channels_ * width_out_  * height_out_);

        int border_x = width_/2 - width_out_/2;
        int border_y = height_/2 - height_out_/2;

        border_x = border_x > 0 ? border_x : 0;
        border_y = border_y > 0 ? border_y : 0;

        top_mat.setTo(0);

        for (int f_offset = 0; f_offset < conv_out_channels_; f_offset+=F_BATCH) {
            for (int s_offset = 0; s_offset < conv_in_channels_; s_offset+=S_BATCH) {
                for (int ff = 0; ff < F_BATCH; ff++) {
                    for (int ss = 0; ss < S_BATCH; ss++) {
                        int f = f_offset + ff;
                        int s = s_offset + ss;

                        int access_f_offset = f * height_out_;
                        int access_s_offset = s * height_;

                        for (int g = 0; g < NUM_GAUSS; ++g) {
                            int param_offset = -1;
                            if (INPUT_FORMAT == FastGaussForward<float>::SGF)
                                param_offset = OFFSET(0, s,g,f, 1, conv_in_channels_, NUM_GAUSS, conv_out_channels_);
                            else if (INPUT_FORMAT == FastGaussForward<float>::FGS)
                                param_offset = OFFSET(0, f,g,s, 1, conv_out_channels_, NUM_GAUSS, conv_in_channels_);

                            float w = filter_weights[param_offset];

                            float offset_x = filter_offsets_float_mu1[param_offset] - kernel_width/2;
                            float offset_y = filter_offsets_float_mu2[param_offset] - kernel_height/2;

                            int offset_x_int = floor(offset_x);
                            int offset_y_int = floor(offset_y);

                            float interpol_off_x = offset_x - offset_x_int;
                            float interpol_off_y = offset_y - offset_y_int;


                            for (int dy = 0; dy < INTERPOlATION_Dy; ++dy) {
                                for (int dx = 0; dx < INTERPOlATION_Dx; ++dx) {

                                    int access_x_off = offset_x_int + dx;
                                    int access_y_off = offset_y_int + dy;

                                    float interpol_w = w;

                                    interpol_w *= (dx == 0 ? (1-interpol_off_x) : interpol_off_x);
                                    interpol_w *= (dy == 0 ? (1-interpol_off_y) : interpol_off_y);

                                    cv::Rect interm_roi(border_x+std::max(0, access_x_off),
                                                        border_y+std::max(0, access_y_off) + access_s_offset,
                                                        std::min(width_out_ + access_x_off, width_out_ - access_x_off),
                                                        std::min(height_out_ + access_y_off, height_out_ - access_y_off));

                                    cv::Rect top_roi(std::max(0, -access_x_off),
                                                     std::max(0, -access_y_off) + access_f_offset,
                                                     std::min(width_out_ + access_x_off, width_out_ - access_x_off),
                                                     std::min(height_out_ + access_y_off, height_out_ - access_y_off));

                                    //std::cout << "top_roi: " << top_roi << " interm_roi: " << interm_roi  << std::endl;
                                    if (top_roi.width > 0 && top_roi.height > 0 && interm_roi.width > 0 && interm_roi.height > 0) {
                                        top_mat(top_roi) += interpol_w * interm_mat(interm_roi);

                                        //if (f == 0) {
                                        //    printf("sum of f,s,g=%d,%d,%d is val: ", f,s,g);
                                        //    std::cout << top_mat(top_roi) << " with top roi: " << top_roi  << " and inter roi: " << interm_roi << " and iter val " << interm_mat(interm_roi) << std::endl;
                                        //}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

    template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Forward_cpu(
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
    const Dtype* gauss_kernel = this->get_gaussian_kernel(stream_[0])->cpu_data();

    // get buffers for all parameters that we learn
    const Dtype* filter_weights = this->param_buffer_w_->cpu_data();
    const Dtype* filter_offsets_float_mu1 = this->param_buffer_mu1_->cpu_data();
    const Dtype* filter_offsets_float_mu2 = this->param_buffer_mu2_->cpu_data();

    for (int i = 0; i < bottom.size(); ++i) {

        const Dtype* bottom_data = bottom[i]->mutable_cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();

        Dtype* interm_data = interm_buffer_.mutable_cpu_data();


        // first perform convolutions with gaussian filter (i.e. gaussian blur)

        // make sure col_buffer is big enough
        this->col_buffer_.Reshape(this->prefilter_h_, this->prefilter_w_, this->height_, this->width_);

        Dtype* col_buff = this->col_buffer_.mutable_cpu_data();

        for (int n = 0; n < this->num_ * this->channels_; ++n) {

            im2col_cpu(bottom_data + n * (this->height_* this->width_), 1, this->height_, this->width_,
                       this->prefilter_h_, this->prefilter_w_,
                       this->prefilter_pad_h_, this->prefilter_pad_w_,
                       this->prefilter_stride_h_, this->prefilter_stride_w_,
                       1,1, col_buff);

            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1 , this->height_ * this->width_, this->prefilter_h_ * this->prefilter_w_,
                                  (Dtype)1., gauss_kernel , col_buff,
                                  (Dtype)0., interm_data + n * this->width_ * this->height_);
        }

        //Dtype* interm_data = bottom[i]->mutable_cpu_data();

        // now we take the blured input data and perform sum over shifted input data with our custom kernel
        offset_and_sum_opencv(interm_data,filter_weights, filter_offsets_float_mu1, filter_offsets_float_mu2,
                            top_data,
                            this->num_, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_,
                            this->width_, this->height_,
                            this->width_out_, this->height_out_,
                            this->kernel_w_, this->kernel_h_);

        // add bias if needed
        if (this->bias_term_) {
            const Dtype *bias = this->param_buffer_bias_->cpu_data();
            for (int n = 0; n < this->num_ ; ++n) {
                this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }
}


template <typename Dtype>
void offset_and_dot_opencv(const Dtype* input_data, const Dtype* error_data,
                           const Dtype* filter_weights, const Dtype* filter_offsets_float_mu1, const Dtype* filter_offsets_float_mu2,
                           Dtype* output_data,
                           const int num_, const int conv_in_channels_, const int NUM_GAUSS, const int conv_out_channels_,
                           const int width_, const int height_,
                           const int width_out_, const int height_out_, const int kernel_width, const int kernel_height,
                           const bool ignore_edge_gradients, const FastGaussForward<float>::PARAM_FORMAT INPUT_FORMAT = FastGaussForward<float>::SGF) {

    // perform offset and sum over individual outputs
#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

    const int INTERPOlATION_Dx = 2;
    const int INTERPOlATION_Dy = 2;

    const int F_BATCH = 8;
    const int S_BATCH = 1;

    for (int n = 0; n < num_; ++n) {
        //printf("n=%d\n",n);


        cv::Mat interm_mat(conv_in_channels_ * height_,width_, CV_32F, (Dtype*)input_data + n * conv_in_channels_ * width_ * height_);
        cv::Mat top_mat_org(conv_out_channels_ * height_out_, width_out_, CV_32F, (Dtype*)error_data + n * conv_out_channels_ * width_out_  * height_out_);

        // copy top matrix to another buffer so that we do not modify original data
        cv::Mat top_mat;
        top_mat_org.copyTo(top_mat);

        // set right/bottom edges to zero if we should ignore them (for GPU compatability)
        if (ignore_edge_gradients) {
            for (int f = 0; f< conv_out_channels_; ++f) {

                int access_f_offset = f * height_out_;

                bool disable_last_column = false;
                bool disable_last_row = false;

                if (width_out_ >= 64) disable_last_column = width_out_ % 64 == 0 ? true : false;
                else if (width_out_ >= 32) disable_last_column = width_out_ % 32 == 0 ? true : false;
                else if (width_out_ >= 16) disable_last_column = width_out_ % 16 == 0 ? true : false;
                else if (width_out_ >= 8) disable_last_column = width_out_ % 8 == 0 ? true : false;

                if (height_out_ >= 64) disable_last_row = height_out_ % 64 == 0 ? true : false;
                else if (height_out_ >= 32) disable_last_row = height_out_ % 32 == 0 ? true : false;
                else if (height_out_ >= 16) disable_last_row = height_out_ % 16 == 0 ? true : false;
                else if (height_out_ >= 8) disable_last_row = height_out_ % 8 == 0 ? true : false;

                if (disable_last_column) top_mat(cv::Rect(width_out_-1, access_f_offset, 1, height_out_ )) = 0.0f;
                if (disable_last_row) top_mat(cv::Rect(0, height_out_-1 + access_f_offset , width_out_, 1)) = 0.0f;
            }
        }
        for (int f_offset = 0; f_offset < conv_out_channels_; f_offset+=F_BATCH) {
            for (int s_offset = 0; s_offset < conv_in_channels_; s_offset+=S_BATCH) {
                for (int ff = 0; ff < F_BATCH; ff++) {
                    for (int ss = 0; ss < S_BATCH; ss++) {
                        int f = f_offset + ff;
                        int s = s_offset + ss;

                        int access_f_offset = f * height_out_;
                        int access_s_offset = s * height_;

                        for (int g = 0; g < NUM_GAUSS; ++g) {

                            int param_output_offset = OFFSET(0, s,g,f, 1, conv_in_channels_, NUM_GAUSS, conv_out_channels_);

                            int param_offset = -1;
                            if (INPUT_FORMAT == FastGaussForward<float>::SGF)
                                param_offset = OFFSET(0, s,g,f, 1, conv_in_channels_, NUM_GAUSS, conv_out_channels_);
                            else if (INPUT_FORMAT == FastGaussForward<float>::FGS)
                                param_offset = OFFSET(0, f,g,s, 1, conv_out_channels_, NUM_GAUSS, conv_in_channels_);

                            float w = filter_weights[param_offset];

                            float offset_x = filter_offsets_float_mu1[param_offset] - kernel_width/2;
                            float offset_y = filter_offsets_float_mu2[param_offset] - kernel_height/2;

                            int offset_x_int = floor(offset_x);
                            int offset_y_int = floor(offset_y);

                            float interpol_off_x = offset_x - offset_x_int;
                            float interpol_off_y = offset_y - offset_y_int;

                            for (int dy = 0; dy < INTERPOlATION_Dy; ++dy) {
                                for (int dx = 0; dx < INTERPOlATION_Dx; ++dx) {

                                    int access_x_off = offset_x_int + dx;
                                    int access_y_off = offset_y_int + dy;

                                    float interpol_w = 1;

                                    interpol_w *= (dx == 0 ? (1-interpol_off_x) : interpol_off_x);
                                    interpol_w *= (dy == 0 ? (1-interpol_off_y) : interpol_off_y);

                                    cv::Rect interm_roi(std::max(0, access_x_off),
                                                        std::max(0, access_y_off) + access_s_offset,
                                                        std::min(width_out_ + access_x_off, width_out_ - access_x_off),
                                                        std::min(height_out_ + access_y_off, height_out_ - access_y_off));

                                    cv::Rect top_roi(std::max(0, -access_x_off),
                                                     std::max(0, -access_y_off) + access_f_offset,
                                                     std::min(width_out_ + access_x_off, width_out_ - access_x_off),
                                                     std::min(height_out_ + access_y_off, height_out_ - access_y_off));


                                    if (top_roi.width > 0 && top_roi.height > 0 && interm_roi.width > 0 && interm_roi.height > 0) {

                                        output_data[param_output_offset] += top_mat(top_roi).dot(interpol_w * interm_mat(interm_roi));

                                        /*if (f == 0 && s == 0 && g == 0)
                                        {
                                            printf("sum of f,s,g=%d,%d,%d, n=%d from ", f,s,g, n);
                                            std::cout << "sum " << output_data[param_output_offset] << " and top " << top_mat(top_roi) << " with top roi: " << top_roi  << " and inter roi: " << interm_roi << " and inter val " << interm_mat(interm_roi) << " and w: " << interpol_w << std::endl;
                                        }*/
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
// get buffers for all parameters that we learn
    const Dtype* filter_weights = this->param_buffer_w_->cpu_data();
    const Dtype* filter_offsets_float_mu1 = this->param_buffer_mu1_->cpu_data();
    const Dtype* filter_offsets_float_mu2 = this->param_buffer_mu2_->cpu_data();


    Dtype* param_weights_diff = this->param_buffer_w_->mutable_cpu_diff();
    Dtype* param_mu1_diff = this->param_buffer_mu1_->mutable_cpu_diff();
    Dtype* param_mu2_diff = this->param_buffer_mu2_->mutable_cpu_diff();
    Dtype* param_sigma_diff = this->param_buffer_sigma_->mutable_cpu_diff();

    Dtype* bias_diff = this->param_buffer_bias_->mutable_cpu_diff();

    Dtype* bwd_gradients_data = this->bwd_gradients.mutable_cpu_data();

    // get filter for gaussian blur step
    const Dtype* deriv_w_kernel = this->get_deriv_kernel_weight(stream_[0])->cpu_data();
    const Dtype* deriv_mu1_kernel = this->get_deriv_kernel_mu1(stream_[0])->cpu_data();
    const Dtype* deriv_mu2_kernel = this->get_deriv_kernel_mu2(stream_[0])->cpu_data();
    const Dtype* deriv_sigma_kernel = this->get_deriv_kernel_sigma(stream_[0])->cpu_data();

    const Dtype* deriv_error_kernel = this->get_deriv_kernel_error(stream_[0])->cpu_data();

    // copy all four kernels into a single blob
    Dtype* deriv_kernels_data  = prefilter_deriv_kernels_.mutable_cpu_data();

    const int prefilter_size = this->prefilter_h_ * this->prefilter_w_ ;

    if (NUM_K > 0) caffe_copy(prefilter_size, deriv_w_kernel, deriv_kernels_data + 0 * prefilter_size);
    if (NUM_K > 1) caffe_copy(prefilter_size, deriv_mu1_kernel, deriv_kernels_data + 1 * prefilter_size);
    if (NUM_K > 2) caffe_copy(prefilter_size, deriv_mu2_kernel, deriv_kernels_data + 2 * prefilter_size);
    if (NUM_K > 3) caffe_copy(prefilter_size, deriv_sigma_kernel, deriv_kernels_data + 3 * prefilter_size);

    // intermediate data for blurred input
    Dtype* interm_data = interm_buffer_.mutable_cpu_data();

    // transform all four accumulated gradients into seperabe buffers of size [S x G x F]
    int param_size = this->NUM_GAUSS * this->conv_in_channels_ * this->conv_out_channels_;

    for (int i = 0; i < bottom.size(); ++i) {

        // input data
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* bottom_error = bottom[i]->mutable_cpu_diff();

        // actual output data
        const Dtype* top_data = top[i]->cpu_data();
        const Dtype* top_error = top[i]->cpu_diff();


        // perform back-propagation of the error values first (we may override errors at boundries to make compatible with GPU version)
        if (propagate_down[i]) {
            // we need to do pre-filtering of the error values

            // make sure col_buffer is big enough
            this->col_buffer_.Reshape(this->prefilter_h_, this->prefilter_w_, this->height_, this->width_);

            Dtype* col_buff = this->col_buffer_.mutable_cpu_data();

            int border_x = this->width_/2 - this->width_out_/2;
            int border_y = this->height_/2 - this->height_out_/2;

            border_x = border_x > 0 ? border_x : 0;
            border_y = border_y > 0 ? border_y : 0;

            // over all top errors where each output channel is considered individual sample as well
            for (int n = 0; n < this->num_ * this->num_output_; ++n) {

                im2col_cpu(top_error + n * (this->height_out_* this->width_out_), 1, this->height_out_, this->width_out_,
                           this->prefilter_h_, this->prefilter_w_,
                           this->prefilter_pad_h_ + border_y, this->prefilter_pad_w_ + border_x,
                           this->prefilter_stride_h_, this->prefilter_stride_w_,
                           1,1, col_buff);

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1 , this->height_ * this->width_, this->prefilter_h_ * this->prefilter_w_,
                                      (Dtype)1., deriv_error_kernel, col_buff,
                                      (Dtype)0., interm_data + n * this->width_ * this->height_ );
            }

            // then use our custom kernel for forwarding, however we need to transpose kernels, which in our case means
            // that we need to rotate mu1,mu2 locations

            // we can re-use bwd_gradients_data buffer for mu1 and mu2 that are rotated
            Dtype *param_mu1_backprop = this->tmp_param_buffer_.mutable_cpu_data() + 0 * param_size;
            Dtype *param_mu2_backprop = this->tmp_param_buffer_.mutable_cpu_data() + 1 * param_size;

            // rot(mu) = (kernel_w-1) - mu
            {
                caffe_copy(param_size, filter_offsets_float_mu1, param_mu1_backprop);
                caffe_copy(param_size, filter_offsets_float_mu2, param_mu2_backprop);

                caffe_scal(param_size, (Dtype)-1, param_mu1_backprop);
                caffe_scal(param_size, (Dtype)-1, param_mu2_backprop);

                caffe_add_scalar(param_size, (Dtype)(this->kernel_w_ - 1), param_mu1_backprop);
                caffe_add_scalar(param_size, (Dtype)(this->kernel_h_ - 1), param_mu2_backprop);
            }

            // now we take the blured error data and perform sum over shifted input data with our custom kernel i.e. forward pass
            offset_and_sum_opencv(interm_data,
                                  filter_weights, param_mu1_backprop, param_mu2_backprop,
                                  bottom_error,
                                  this->num_, this->num_output_, this->NUM_GAUSS, this->channels_,
                                  this->width_, this->height_,
                                  this->width_, this->height_, this->kernel_w_, this->kernel_h_, FastGaussForward<float>::FGS);


        }
        // Gradient w.r.t. bias.
        if (this->bias_term_ && this->param_propagate_down_[1]) {

            Dtype* bias_diff = this->param_buffer_bias_->mutable_cpu_diff();
            for (int n = 0; n < this->num_; ++n) {
                this->backward_cpu_bias(bias_diff, top_error + n * this->top_dim_);
            }
        }

        // Gradient w.r.t w,mu1,mu2 and sigma
        if (this->param_propagate_down_[0]) {

            // first pre-filter input data with appropriate derivative filters
            int size_batch_k = this->num_ * this->channels_ * this->width_ * this->height_;

            // make sure col_buffer is big enough
            this->col_buffer_.Reshape(this->prefilter_h_, this->prefilter_w_, this->height_, this->width_);

            Dtype* col_buff = this->col_buffer_.mutable_cpu_data();

            for (int n = 0; n < this->num_ * this->channels_; ++n) {

                im2col_cpu(bottom_data + n * (this->height_* this->width_), 1, this->height_, this->width_,
                           this->prefilter_h_, this->prefilter_w_,
                           this->prefilter_pad_h_, this->prefilter_pad_w_,
                           this->prefilter_stride_h_, this->prefilter_stride_w_,
                           1,1, col_buff);

                for (int k = 0; k < this->NUM_K; ++k) {
                    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1 , this->height_ * this->width_, this->prefilter_h_ * this->prefilter_w_,
                                          (Dtype)1., deriv_kernels_data + k * (this->prefilter_h_ * this->prefilter_w_) , col_buff,
                                          (Dtype)0., interm_data + n * this->width_ * this->height_ + k * size_batch_k);
                }
            }
            Blob<Dtype> top_error_expended;
            Dtype* top_error_ex = (Dtype*)top_error;

            if (this->width_out_ != this->width_ && this->height_out_ != this->height_) {
                // extend top data if we have top data of not the same size

                int border_x = this->width_/2 - this->width_out_/2;
                int border_y = this->height_/2 - this->height_out_/2;

                border_x = border_x > 0 ? border_x : 0;
                border_y = border_y > 0 ? border_y : 0;

                top_error_expended.Reshape(top[i]->num(), top[i]->channels(), top[i]->height() + 2*border_y, top[i]->width() + 2*border_x);

                top_error_ex = top_error_expended.mutable_cpu_data();

                memset(top_error_ex, 0, sizeof(Dtype) * top_error_expended.count());

                for (int n = 0; n < top_error_expended.num(); ++n) {
                    for (int c = 0; c < top_error_expended.channels(); ++c) {
                        for (int h = 0; h < top[i]->height(); ++h) {
                            for (int w = 0; w < top[i]->width(); ++w) {
                                top_error_ex[top_error_expended.offset(n,c,border_y+h,border_x+w)] = top_error[top[i]->offset(n,c,h,w)];
                            }
                        }
                    }
                }

            }

            // then collect gradients by shifting convolved bottom input data and multiplying it with the top error data
            for (int k = 0; k < this->NUM_K; ++k) {
                //printf("k=%d\n",k);

                offset_and_dot_opencv(interm_data + k * size_batch_k,
                                      top_error_ex,
                                      filter_weights, filter_offsets_float_mu1, filter_offsets_float_mu2,
                                      bwd_gradients_data + k * param_size,
                                      this->num_, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_,
                                      this->width_, this->height_,
                                      this->width_, this->height_, this->kernel_w_, this->kernel_h_, this->ignore_edge_gradients_);

            }
        }


    }

    // we need accumulate gradients them to the final buffer and add weights to some derivates
    if (this->param_propagate_down_[0]) {
        // multiply gradients with appropriate weights
        /// add add weight multiplyer as specifed by derivative formula only for mu1,mu2 and sigma
        if (NUM_K > 1) caffe_mul(param_size, bwd_gradients_data + 1 * param_size, filter_weights, bwd_gradients_data + 1 * param_size); // mu1
        if (NUM_K > 2) caffe_mul(param_size, bwd_gradients_data + 2 * param_size, filter_weights, bwd_gradients_data + 2 * param_size); // mu2
        if (NUM_K > 3) caffe_mul(param_size, bwd_gradients_data + 3 * param_size, filter_weights, bwd_gradients_data + 3 * param_size); // sigma

        // for weight gradient we only accumulate to final buffer
        if (NUM_K > 0) caffe_axpy(param_size, (Dtype)1, bwd_gradients_data + 0 * param_size, param_weights_diff); // w
        if (NUM_K > 1) caffe_axpy(param_size, (Dtype)1, bwd_gradients_data + 1 * param_size, param_mu1_diff); // mu1
        if (NUM_K > 2) caffe_axpy(param_size, (Dtype)1, bwd_gradients_data + 2 * param_size, param_mu2_diff); // mu2
        if (NUM_K > 3) caffe_axpy(param_size, (Dtype)1, bwd_gradients_data + 3 * param_size, param_sigma_diff); // sigma

        // if we need to ignore last few gauss then make sure we do not update their parameters
        if (this->num_gauss_ignore > 0) {
            this->set_last_n_gauss_to_zero(param_weights_diff, this->num_gauss_ignore);
            this->set_last_n_gauss_to_zero(param_mu1_diff, this->num_gauss_ignore);
            this->set_last_n_gauss_to_zero(param_mu2_diff, this->num_gauss_ignore);
            this->set_last_n_gauss_to_zero(param_sigma_diff, this->num_gauss_ignore);
        }
    }
}

INSTANTIATE_CLASS(FastAproxGaussianConvLayer);

}   // namespace caffe
