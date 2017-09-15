#ifdef USE_CUDNN
#include <algorithm>
#include <vector>
#include <caffe/layers/gauss_conv_layer.hpp>

//#include "caffe/vision_layers.hpp"
#include "caffe/layers/gauss_conv_layer.hpp"
#include "caffe/util/math_functions_extra.hpp"

#include "caffe/util/fast_gauss_forward.hpp"
#include "caffe/util/fast_gauss_backward.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

namespace caffe {

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    BaseGaussianConvLayer<Dtype>::LayerSetUp(bottom, top);

    CHECK_EQ(1, this->group_)
        << "CuDNNGaussianConvLayer does not support group parameter at the moment";

    // Initialize CUDA streams and cuDNN.
    stream_         = new cudaStream_t[this->group_];
    handle_         = new cudnnHandle_t[this->group_ ];


    // Initialize algorithm arrays
    fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
    bwd_data_algo_  = new cudnnConvolutionFwdAlgo_t[bottom.size()];
    bwd_error_algo_ = new cudnnConvolutionFwdAlgo_t[bottom.size()];

    // initialize size arrays
    workspace_fwd_sizes_ = new size_t[bottom.size()];
    workspace_bwd_data_sizes_ = new size_t[bottom.size()];
    workspace_bwd_error_sizes_ = new size_t[bottom.size()];

    // workspace data
    workspaceSizeInBytes = 0;
    workspaceData = NULL;
    workspace = new void*[this->group_ ];

    for (size_t i = 0; i < bottom.size(); ++i) {
        // initialize all to default algorithms
        fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
        bwd_data_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
        bwd_error_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
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

    for (int g = 0; g < this->group_ ; g++) {
        CUDA_CHECK(cudaStreamCreate(&stream_[g]));
        CUDNN_CHECK(cudnnCreate(&handle_[g]));

        CUDNN_CHECK (cudnnSetStream(handle_[g], stream_[g]));

        workspace[g] = NULL;
    }

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

    // TODO: read kernel sizes from prototxt or calculate it from sigma sizes !!!
    gauss_kernel_h_ = 7;
    gauss_kernel_w_ = 7;

    gauss_kernel_pad_h_ = 3;
    gauss_kernel_pad_w_ = 3;

    gauss_kernel_stride_h_ = 1;
    gauss_kernel_stride_w_ = 1;

    cudnn::createFilterDesc<Dtype>(&fwd_g_kernel_desc_,
                                   1, 1, gauss_kernel_h_, gauss_kernel_w_);

    cudnn::createFilterDesc<Dtype>(&bwd_g_kernel_desc_,
                                   NUM_K, 1, gauss_kernel_h_, gauss_kernel_w_);

    this->use_interpolation_ = true;

    // create buffers used to generate kernels (i.e. we need only one kernel
    gauss_param_prefilter_w_.Reshape(1,1,1,1);
    gauss_param_prefilter_mu1_.Reshape(1,1,1,1);
    gauss_param_prefilter_mu2_.Reshape(1,1,1,1);
    gauss_param_prefilter_sigma_.Reshape(1,1,1,1);

    // by default we generate kernels with w=1, mu=(0,0) so fill buffers with them
    // NOTE: mu=(0,0) is center of kernel so use that value
    *(gauss_param_prefilter_w_.mutable_cpu_data()) = 1.0f;
    *(gauss_param_prefilter_mu1_.mutable_cpu_data()) = gauss_kernel_h_/2;
    *(gauss_param_prefilter_mu2_.mutable_cpu_data()) = gauss_kernel_h_/2;

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

    handles_setup_ = true;

    paralel_streams = new cudaStream_t[4];
    for (int g = 0; g < 4; ++g) {
        cudaStreamCreate(&paralel_streams[g]);
    }

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

    // use inter buffer for both fwd and bwd passes so allocate buffer with suitable size for both
    interm_buffer_.Reshape(this->num_, this->channels_* 4, height, width);

    this->gaussian_kernel_.Reshape(1,1,this->gauss_kernel_h_, this->gauss_kernel_w_);

    this->deriv_kernel_weight_.ReshapeLike(this->gaussian_kernel_);
    this->deriv_kernel_mu1_.ReshapeLike(this->gaussian_kernel_);
    this->deriv_kernel_mu2_.ReshapeLike(this->gaussian_kernel_);
    this->deriv_kernel_sigma_.ReshapeLike(this->gaussian_kernel_);
    this->deriv_kernel_error_.ReshapeLike(this->gaussian_kernel_);

    this->deriv_kernels_.Reshape(1,NUM_K,this->gauss_kernel_h_, this->gauss_kernel_w_);


    this->bwd_gradients.Reshape(4, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_);

    // temporary buffer used during the back-propagation of the error where we rotate mu1 and mu2
    this->tmp_param_buffer_.Reshape(2, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_);

    for (int i = 0; i < bottom.size(); i++) {

        cudnn::setTensor4dDesc<Dtype>(&top_bias_descs_[i],
                                      this->num_ ,
                                      this->num_output_, height_out, width_out,
                                      this->num_output_ * height_out * width_out,
                                      height_out * width_out, width_out, 1);

        cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
                                      this->num_ *this->num_output_, 1, height, width,
                                      1 * height * width,
                                      height * width, width, 1);

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
                                             fwd_g_kernel_desc_, gauss_kernel_pad_h_, gauss_kernel_pad_w_,
                                             gauss_kernel_stride_h_, gauss_kernel_stride_w_);

            // choose forward and backward algorithms + workspace(s)
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
                                                            bottom_descs_[i],
                                                            fwd_g_kernel_desc_,
                                                            fwd_conv_descs_[i],
                                                            fwd_interm_descs_[i],
                                                            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                            workspace_limit_bytes,
                                                            &fwd_algo_[i]));

            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
                                                                bottom_descs_[i],
                                                                fwd_g_kernel_desc_,
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
                                             bwd_g_kernel_desc_, gauss_kernel_pad_h_, gauss_kernel_pad_w_,
                                             gauss_kernel_stride_h_, gauss_kernel_stride_w_);

            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
                                                            bottom_descs_[i],
                                                            bwd_g_kernel_desc_,
                                                            bwd_conv_data_descs_[i],
                                                            bwd_interm_data_descs_[i],
                                                            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                            workspace_limit_bytes,
                                                            &bwd_data_algo_[i]));

            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
                                                                bottom_descs_[i],
                                                                bwd_g_kernel_desc_,
                                                                bwd_conv_data_descs_[i],
                                                                bwd_interm_data_descs_[i],
                                                                bwd_data_algo_[i],
                                                                &(workspace_bwd_data_sizes_[i])));

            // descriptor and algo for pre-filetring of error data with reversed convolution filter

            cudnn::setTensor4dDesc<Dtype>(&bwd_interm_error_descs_[i],
                                          this->num_ * this->num_output_, 1, height, width,
                                          1 * height * width,
                                          height * width, width, 1);

            cudnn::setConvolutionDesc<Dtype>(&bwd_conv_error_descs_[i], top_descs_[i],
                                             bwd_g_kernel_desc_, gauss_kernel_pad_h_, gauss_kernel_pad_w_,
                                             gauss_kernel_stride_h_, gauss_kernel_stride_w_);

            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
                                                            top_descs_[i],
                                                            bwd_g_kernel_desc_,
                                                            bwd_conv_error_descs_[i],
                                                            bwd_interm_error_descs_[i],
                                                            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                            workspace_limit_bytes,
                                                            &bwd_error_algo_[i]));

            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
                                                                top_descs_[i],
                                                                bwd_g_kernel_desc_,
                                                                bwd_conv_data_descs_[i],
                                                                bwd_interm_error_descs_[i],
                                                                bwd_error_algo_[i],
                                                                &(workspace_bwd_error_sizes_[i])));

        }
    }


    // check how much memory do we need for our custom kernels
    caffe::fast_gauss_forward<Dtype>(NULL, NULL, NULL, NULL, FAST_GAUSS_PARAM_SGF, NULL,
                                     this->num_, this->channels_, this->num_output_, this->NUM_GAUSS,
                                     this->width_out_, this->height_out_,
                                     this->kernel_w_, this->kernel_h_,
                                     this->use_interpolation_,
                                     NULL,&buffer_fwd_.filtered_images_sizes_,
                                     NULL,&buffer_fwd_.filter_weights_sizes_,
                                     NULL,&buffer_fwd_.filter_offsets_sizes_,
                                     NULL);

    caffe::fast_gauss_backward_multi_subfeatures<Dtype>(NULL, NULL, NULL, NULL, NULL, NULL,
                                                        this->num_, this->channels_, this->num_output_, this->NUM_GAUSS, NUM_K,
                                                        this->width_out_, this->height_out_,
                                                        this->kernel_w_, this->kernel_h_,
                                                        this->use_interpolation_, this->ignore_edge_gradients_,
                                                        NULL,&buffer_bwd_.filtered_images_sizes_,
                                                        NULL,&buffer_bwd_.error_image_sizes_,
                                                        NULL,&buffer_bwd_.filter_weights_sizes_,
                                                        NULL,&buffer_bwd_.filter_offsets_sizes_);

    // reduce over all workspace sizes to get a maximum to allocate / reallocate
    size_t total_workspace_fwd = 0;
    size_t total_workspace_bwd_data = 0;
    size_t total_workspace_bwd_error = 0;

    for (size_t i = 0; i < bottom.size(); i++) {
        total_workspace_fwd        = std::max(total_workspace_fwd,
                                              workspace_fwd_sizes_[i]);
        total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                              workspace_bwd_data_sizes_[i]);
        total_workspace_bwd_error   = std::max(total_workspace_bwd_error,
                                              workspace_bwd_error_sizes_[i]);
    }

    total_workspace_fwd         = std::max(total_workspace_fwd,
                                          buffer_fwd_.filtered_images_sizes_ +
                                          buffer_fwd_.filter_weights_sizes_ +
                                          buffer_fwd_.filter_offsets_sizes_);
    total_workspace_bwd_data    = std::max(total_workspace_bwd_data,
                                           buffer_bwd_.filtered_images_sizes_ +
                                           buffer_bwd_.error_image_sizes_ +
                                           buffer_bwd_.filter_weights_sizes_ +
                                           buffer_bwd_.filter_offsets_sizes_);

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
                fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
                bwd_data_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
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

        // TODO: make all memory align to 4x 32bit values
        buffer_fwd_.filtered_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData));
        buffer_fwd_.filter_offsets_and_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_);
        buffer_fwd_.filter_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_);
        buffer_fwd_.filter_offsets = reinterpret_cast<int*>(reinterpret_cast<char *>(workspaceData) + buffer_fwd_.filtered_images_sizes_ + buffer_fwd_.filter_weights_sizes_);


        buffer_bwd_.filtered_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData));
        buffer_bwd_.error_images = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_);
        buffer_bwd_.filter_weights = reinterpret_cast<Dtype*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_ + buffer_bwd_.error_image_sizes_);
        buffer_bwd_.filter_offsets = reinterpret_cast<int*>(reinterpret_cast<char *>(workspaceData) + buffer_bwd_.filtered_images_sizes_ + buffer_bwd_.error_image_sizes_ + buffer_bwd_.filter_weights_sizes_);
    }

    // Tensor descriptor for bias.
    if (this->bias_term_) {
        cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
                                      1, this->num_output_ , 1, 1);
    }
}

template <typename Dtype>
FastAproxGaussianConvLayer<Dtype>::~FastAproxGaussianConvLayer() {

}


template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::test_kernel_cpu(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
								const float* filter_weights, float* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height, const bool use_interpolation) {

	//caffe::fast_gauss_forward<float>(filtered_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
}


template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::test_kernel_gpu(const Dtype* filtered_images, const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
								const Dtype* filter_weights, Dtype* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height, const bool use_interpolation) {

    Dtype* prepared_filtered_images;
    Dtype* prepared_filter_weights;
    int* prepared_filter_offsets;
    Dtype* prepared_filter_offsets_and_weights;

    size_t prepared_filtered_images_size,
            prepared_filter_weights_size,
            prepared_filter_offsets_size;

    // call first with valid pointers to input sizes to get size of buffer that we need to allocate
    caffe::fast_gauss_forward<Dtype>(filtered_images,
                                    filter_offsets_float_x, filter_offsets_float_y, filter_weights, FAST_GAUSS_PARAM_SGF,
                                    output,
                                    I, S, F, G,
                                    img_width, img_height,
                                    kernel_width, kernel_height,
                                    use_interpolation,
                                    NULL,&prepared_filtered_images_size,
                                    NULL,&prepared_filter_weights_size,
                                    NULL,&prepared_filter_offsets_size,
                                    NULL);

    CUDA_CHECK(cudaMalloc(&prepared_filtered_images, prepared_filtered_images_size));
    CUDA_CHECK(cudaMemset(prepared_filtered_images, 0,  prepared_filtered_images_size));

    CUDA_CHECK(cudaMalloc(&prepared_filter_weights, prepared_filter_weights_size));

    CUDA_CHECK(cudaMalloc(&prepared_filter_offsets, prepared_filter_offsets_size));
    CUDA_CHECK(cudaMemset(prepared_filter_offsets,0, prepared_filter_offsets_size));

	CUDA_CHECK(cudaMalloc(&prepared_filter_offsets_and_weights, prepared_filter_weights_size+prepared_filter_offsets_size));
	CUDA_CHECK(cudaMemset(prepared_filter_offsets_and_weights,0, prepared_filter_weights_size+prepared_filter_offsets_size));


	for (int i = 0; i < 1; ++i) {
		cudaDeviceSynchronize();

		clock_t start_t = clock();
		caffe::fast_gauss_forward<Dtype>(filtered_images,
										 filter_offsets_float_x, filter_offsets_float_y, filter_weights, FAST_GAUSS_PARAM_SGF,
                                         output,
										 I, S, F, G,
										 img_width, img_height,
										 kernel_width, kernel_height,
										 use_interpolation,
										 prepared_filtered_images,0,
										 NULL,0,
										 NULL,0,
										 prepared_filter_offsets_and_weights);
		cudaDeviceSynchronize();
		clock_t end_t = clock();

		std::cout << "fast_gauss_forward in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
	}
    cudaFree(prepared_filter_weights);
    cudaFree(prepared_filter_offsets);
    cudaFree(prepared_filter_offsets_and_weights);

    cudaFree(prepared_filtered_images);


}

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::test_backward_kernel_gpu(const float* filtered_images, const float* error_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
                                                        const float* filter_weights, float* output,
                                                        const int I, const int S, const int F, const int G,
                                                        const int img_width, const int img_height,
                                                        const int kernel_width, const int kernel_height) {


    caffe::fast_gauss_backward<float>(filtered_images, error_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
}


template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::test_backward_multi_subfeature_kernel_gpu(const float* filtered_images, const float* error_images, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
																 const float* filter_weights, float* output,
																 const int K, const int I, const int S, const int F, const int G,
																 const int img_width, const int img_height,
																 const int kernel_width, const int kernel_height, const bool use_interpolation, const bool ignore_edge_gradients) {
	float* prepared_filtered_images;
	float* prepared_error_images;
	float* prepared_filter_weights;
	int* prepared_filter_offsets;

	size_t prepared_filtered_images_size,
			prepared_error_images_size,
			prepared_filter_weights_size,
			prepared_filter_offsets_size;

	// call first with valid pointers to input sizes to get size of buffer that we need to allocate
	caffe::fast_gauss_backward_multi_subfeatures<float>(filtered_images, error_images,
														filter_offsets_float_x, filter_offsets_float_y,
														filter_weights, output,
														I, S, F, G, K,
														img_width, img_height,
														kernel_width, kernel_height,
														use_interpolation, ignore_edge_gradients,
														0,&prepared_filtered_images_size,
														0,&prepared_error_images_size,
														0,&prepared_filter_weights_size,
														0,&prepared_filter_offsets_size);


	CUDA_CHECK(cudaMalloc(&prepared_filtered_images, prepared_filtered_images_size));
	CUDA_CHECK(cudaMemset(prepared_filtered_images, 0,  prepared_filtered_images_size));

	CUDA_CHECK(cudaMalloc(&prepared_error_images, prepared_error_images_size));
	CUDA_CHECK(cudaMemset(prepared_error_images, 0,  prepared_error_images_size));

    CUDA_CHECK(cudaMalloc(&prepared_filter_weights, prepared_filter_weights_size));

    CUDA_CHECK(cudaMalloc(&prepared_filter_offsets, prepared_filter_offsets_size));
    CUDA_CHECK(cudaMemset(prepared_filter_offsets,0, prepared_filter_offsets_size));

	for (int i = 0; i < 30; ++i) {
		CUDA_CHECK(cudaMemset(output,0, sizeof(float) * S* F* G* K));
		cudaDeviceSynchronize();

		clock_t start_t = clock();

		caffe::fast_gauss_backward_multi_subfeatures<float>(filtered_images, error_images,
															filter_offsets_float_x, filter_offsets_float_y,
															filter_weights, output,
															I, S, F, G, K,
															img_width, img_height,
															kernel_width, kernel_height,
															use_interpolation, ignore_edge_gradients,
															prepared_filtered_images,0,
															prepared_error_images,0,
															prepared_filter_weights,0,
															prepared_filter_offsets,0);
		cudaDeviceSynchronize();
		clock_t end_t = clock();

		std::cout << "fast_gauss_backward_multi_subfeatures in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
	}
	cudaFree(prepared_filter_weights);
	cudaFree(prepared_filter_offsets);

	cudaFree(prepared_filtered_images);
    cudaFree(prepared_error_images);

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

    if (std::fabs(gaussian_kernel_variance - *sigma_cpu) > 1e-5) {
        // we compute kernels for blur using the same code as in std-implementation but we compute only for a single
        // component i.e., num_in_channels = 1, num_out_channels = 1, num_gauss = 1, and we use weight=1, mu = [0,0]

        bool is_backward_pass = true;
        bool gmm_discretize_mean = false;
        Dtype gmm_component_border_bound = 0;

        this->do_precompute_guassian_weights_gpu(this->gauss_param_prefilter_w_,
                                                 this->gauss_param_prefilter_mu1_,
                                                 this->gauss_param_prefilter_mu2_,
                                                 *this->param_buffer_sigma_.get(),
                                                 1, 1, 1,
                                                 this->gauss_kernel_h_, this->gauss_kernel_w_,
                                                 is_backward_pass,
                                                 this->use_gmm_weight_normalization,
                                                 this->use_gmm_square_gauss_normalization,
                                                 gmm_discretize_mean,
                                                 this->gmm_sigma_lower_bound,
                                                 gmm_component_border_bound,
                                                 &this->gaussian_kernel_,
                                                 NULL, NULL, NULL,
                                                 &this->deriv_kernel_error_,
                                                 &this->deriv_kernel_weight_,
                                                 &this->deriv_kernel_mu1_,
                                                 &this->deriv_kernel_mu2_,
                                                 &this->deriv_kernel_sigma_);

        this->gaussian_kernel_variance = sigma_cpu[0];


        //for debug write kernel with 1 only at center i.e. identity convolution kernel
        {
            Dtype*  gauss_kernel = this->gaussian_kernel_.mutable_cpu_data();
            Dtype*  deriv_weight_kernel = this->deriv_kernel_weight_.mutable_cpu_data();
            Dtype*  deriv_mu1_kernel = this->deriv_kernel_mu1_.mutable_cpu_data();
            Dtype*  deriv_mu2_kernel = this->deriv_kernel_mu2_.mutable_cpu_data();
            Dtype*  deriv_sigma_kernel = this->deriv_kernel_sigma_.mutable_cpu_data();
            Dtype*  deriv_error_kernel = this->deriv_kernel_error_.mutable_cpu_data();


            int h_half = gauss_kernel_h_/2;
            int w_half = gauss_kernel_h_/2;
            int index = 0;
            for (int j = -h_half; j <= h_half; ++j) {
                for (int i = -w_half; i <= w_half; ++i) {

                    Dtype val = i == 0 && j == 0 ? 1 : 0;

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
        plot_blob_data(this->gaussian_kernel_);
        plot_blob_data(this->deriv_kernel_weight_);
        plot_blob_data(this->deriv_kernel_mu1_);
        plot_blob_data(this->deriv_kernel_mu2_);
        plot_blob_data(this->deriv_kernel_sigma_);
        plot_blob_data(this->deriv_kernel_error_);
*/
    }
}

template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_gaussian_kernel(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->gaussian_kernel_;
}

template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_weight(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->deriv_kernel_weight_;
}
template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_mu1(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->deriv_kernel_mu1_;
}
template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_mu2(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->deriv_kernel_mu2_;
}
template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_sigma(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->deriv_kernel_sigma_;
}
template <typename Dtype>
Blob<Dtype>* FastAproxGaussianConvLayer<Dtype>::get_deriv_kernel_error(cudaStream_t stream) {

    update_prefiltering_kernels(stream);

    return &this->deriv_kernel_error_;
}

template <typename Dtype>
void offset_and_sum_opencv(const Dtype* input_data,
                    const Dtype* filter_weights, const Dtype* filter_offsets_float_mu1, const Dtype* filter_offsets_float_mu2,
                    Dtype* output_data,
                    const int num_, const int conv_in_channels_, const int NUM_GAUSS, const int conv_out_channels_,
                    const int width_, const int height_,
                    const int width_out_, const int height_out_, const int INPUT_FORMAT = FAST_GAUSS_PARAM_SGF) {

    // perform offset and sum over individual outputs
#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

    const int INTERPOlATION_Dx = 2;
    const int INTERPOlATION_Dy = 2;

    const int F_BATCH = 4;
    const int S_BATCH = 4;

    for (int n = 0; n < num_; ++n) {
        printf("n=%d\n",n);

        cv::Mat interm_mat(conv_in_channels_ * height_,width_, CV_32F, (Dtype*)input_data + n * conv_in_channels_ * width_ * height_);
        cv::Mat top_mat(conv_out_channels_ * height_out_, width_out_, CV_32F, output_data + n * conv_out_channels_ * width_out_  * height_out_);

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
                            if (INPUT_FORMAT == FAST_GAUSS_PARAM_SGF)
                                param_offset = OFFSET(0, s,g,f, 1, conv_in_channels_, NUM_GAUSS, conv_out_channels_);
                            else if (INPUT_FORMAT == FAST_GAUSS_PARAM_FGS)
                                param_offset = OFFSET(0, f,g,s, 1, conv_out_channels_, NUM_GAUSS, conv_in_channels_);

                            float w = filter_weights[param_offset];

                            float offset_x = filter_offsets_float_mu1[param_offset];
                            float offset_y = filter_offsets_float_mu2[param_offset];

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

                                    cv::Rect interm_roi(std::max(0, access_x_off),
                                                        std::max(0, access_y_off) + access_s_offset,
                                                        std::min(width_ + access_x_off, width_ - access_x_off),
                                                        std::min(height_ + access_y_off, height_ - access_y_off));

                                    cv::Rect top_roi(std::max(0, -access_x_off),
                                                     std::max(0, -access_y_off) + access_f_offset,
                                                     std::min(width_ + access_x_off, width_ - access_x_off),
                                                     std::min(height_ + access_y_off, height_ - access_y_off));

                                    top_mat(top_roi) +=  interpol_w * interm_mat(interm_roi);

                                    /*for (int jj = 0; jj < height_out_; ++jj) {
                                        for (int ii = 0; ii < width_out_; ++ii) {
                                            if (0 <= jj + access_y_off && jj + access_y_off < height_ &&
                                                0 <= ii + access_x_off && ii + access_x_off < width_) {

                                                int in_offset = OFFSET(n, s, jj + access_y_off, ii + access_x_off, num_, conv_in_channels_, height_, width_);
                                                int out_offset = OFFSET(n, f, jj, ii, num_, conv_out_channels_, height_out_, width_out_);

                                                output_data[out_offset] += interpol_w * input_data[in_offset];
                                            }
                                        }
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
    const Dtype* gauss_kernel = this->get_gaussian_kernel(stream_[0])->gpu_data();

    // get buffers for all parameters that we learn
    const Dtype* filter_weights = this->param_buffer_w_->cpu_data();
    const Dtype* filter_offsets_float_mu1 = this->param_buffer_mu1_->cpu_data();
    const Dtype* filter_offsets_float_mu2 = this->param_buffer_mu2_->cpu_data();

    for (int i = 0; i < bottom.size(); ++i) {
        // actual output data
        Dtype* top_data = top[i]->mutable_cpu_data();

        // Forward through cuDNN and our custom kernel
        cudaDeviceSynchronize();
        clock_t start_t = clock();

        // first perform convolutions with gaussian filter (i.e. gaussian blur)
        // we use cudnn forward implementation by casting
        //  - input into [N*S x 1 x HxW]
        //  - weight into [1 x K_h x K_w]
        // this effectively perform single convolution on each input feature and we get
        //  - output [N*S x 1 x HxW] => [N x S x HxW]
        if (0)
        CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
                                            cudnn::dataType<Dtype>::one,
                                            bottom_descs_[i], bottom[i]->gpu_data(),
                                            fwd_g_kernel_desc_, gauss_kernel,
                                            fwd_conv_descs_[i],
                                            fwd_algo_[i], workspace[0], workspace_fwd_sizes_[i],
                                            cudnn::dataType<Dtype>::zero,
                                            fwd_interm_descs_[i], interm_buffer_.mutable_gpu_data()));

        cudaDeviceSynchronize();
        // now we take the blured input data and perform sum over shifted input data with our custom kernel
        //Dtype* interm_data = interm_buffer_.mutable_cpu_data();
        Dtype* interm_data = bottom[i]->mutable_cpu_data();

        offset_and_sum_opencv(interm_data,filter_weights, filter_offsets_float_mu1, filter_offsets_float_mu2,
                            top_data,
                            this->num_, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_,
                            this->width_, this->height_,
                            this->width_out_, this->height_out_);

        // add bias if needed
        if (this->bias_term_) {
            const Dtype* bias_data = this->param_buffer_bias_->gpu_data();
            CUDNN_CHECK(cudnnAddTensor(handle_[0],
                                       cudnn::dataType<Dtype>::one,
                                       bias_desc_, bias_data ,
                                       cudnn::dataType<Dtype>::one,
                                       top_bias_descs_[i], top[i]->mutable_gpu_data()));
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
void offset_and_dot_opencv(const Dtype* input_data, const Dtype* error_data,
                           const Dtype* filter_weights, const Dtype* filter_offsets_float_mu1, const Dtype* filter_offsets_float_mu2,
                           Dtype* output_data,
                           const int num_, const int conv_in_channels_, const int NUM_GAUSS, const int conv_out_channels_,
                           const int width_, const int height_,
                           const int width_out_, const int height_out_, const bool ignore_edge_gradients, const int INPUT_FORMAT = FAST_GAUSS_PARAM_SGF) {

    // perform offset and sum over individual outputs
#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

    const int INTERPOlATION_Dx = 2;
    const int INTERPOlATION_Dy = 2;

    const int F_BATCH = 4;
    const int S_BATCH = 4;

    for (int n = 0; n < num_; ++n) {
        //printf("n=%d\n",n);

        cv::Mat interm_mat(conv_in_channels_ * height_,width_, CV_32F, (Dtype*)input_data + n * conv_in_channels_ * width_ * height_);
        cv::Mat top_mat(conv_out_channels_ * height_out_, width_out_, CV_32F, (Dtype*)error_data + n * conv_out_channels_ * width_out_  * height_out_);

        // set right/bottom edges to zero if we should ignore them (for GPU compatability)
        if (ignore_edge_gradients) {
            for (int f = 0; f< conv_out_channels_; ++f) {

                int access_f_offset = f * height_out_;

                top_mat(cv::Rect(width_out_-1, access_f_offset, 1, height_out_ )) = 0.0f;
                top_mat(cv::Rect(0, height_out_-1 + access_f_offset , width_out_, 1)) = 0.0f;
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
                            if (INPUT_FORMAT == FAST_GAUSS_PARAM_SGF)
                                param_offset = OFFSET(0, s,g,f, 1, conv_in_channels_, NUM_GAUSS, conv_out_channels_);
                            else if (INPUT_FORMAT == FAST_GAUSS_PARAM_FGS)
                                param_offset = OFFSET(0, f,g,s, 1, conv_out_channels_, NUM_GAUSS, conv_in_channels_);

                            float w = filter_weights[param_offset];

                            float offset_x = filter_offsets_float_mu1[param_offset];
                            float offset_y = filter_offsets_float_mu2[param_offset];

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
                                                        std::min(width_ + access_x_off, width_ - access_x_off),
                                                        std::min(height_ + access_y_off, height_ - access_y_off));

                                    cv::Rect top_roi(std::max(0, -access_x_off),
                                                     std::max(0, -access_y_off) + access_f_offset,
                                                     std::min(width_ + access_x_off, width_ - access_x_off),
                                                     std::min(height_ + access_y_off, height_ - access_y_off));



                                    output_data[param_output_offset] += top_mat(top_roi).dot(interpol_w * interm_mat(interm_roi));

                                    /*for (int jj = 0; jj < height_out_; ++jj) {
                                        for (int ii = 0; ii < width_out_; ++ii) {
                                            if (0 <= jj + access_y_off && jj + access_y_off < height_ &&
                                                0 <= ii + access_x_off && ii + access_x_off < width_) {

                                                int in_offset = OFFSET(n, s, jj + access_y_off, ii + access_x_off, num_, conv_in_channels_, height_, width_);
                                                int out_offset = OFFSET(n, f, jj, ii, num_, conv_out_channels_, height_out_, width_out_);

                                                output_data[out_offset] += interpol_w * input_data[in_offset];
                                            }
                                        }
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
    const Dtype* deriv_w_kernel = this->get_deriv_kernel_weight(stream_[0])->gpu_data();
    const Dtype* deriv_mu1_kernel = this->get_deriv_kernel_mu1(stream_[0])->gpu_data();
    const Dtype* deriv_mu2_kernel = this->get_deriv_kernel_mu2(stream_[0])->gpu_data();
    const Dtype* deriv_sigma_kernel = this->get_deriv_kernel_sigma(stream_[0])->gpu_data();

    const Dtype* deriv_error_kernel = this->get_deriv_kernel_error(stream_[0])->gpu_data();

    // copy all four kernels into a single blob
    Dtype* deriv_kernels_data  = deriv_kernels_.mutable_gpu_data();

    const int prefilter_size = this->gauss_kernel_h_ * this->gauss_kernel_w_ ;

    if (NUM_K > 0) caffe_gpu_memcpy(prefilter_size * sizeof(float), deriv_w_kernel, deriv_kernels_data + 0 * prefilter_size);
    if (NUM_K > 1) caffe_gpu_memcpy(prefilter_size * sizeof(float), deriv_mu1_kernel, deriv_kernels_data + 1 * prefilter_size);
    if (NUM_K > 2) caffe_gpu_memcpy(prefilter_size * sizeof(float), deriv_mu2_kernel, deriv_kernels_data + 2 * prefilter_size);
    if (NUM_K > 3) caffe_gpu_memcpy(prefilter_size * sizeof(float), deriv_sigma_kernel, deriv_kernels_data + 3 * prefilter_size);

    // intermediate data for blurred input
    Dtype* interm_data = interm_buffer_.mutable_gpu_data();

    // transform all four accumulated gradients into seperabe buffers of size [S x G x F]
    int param_size = this->NUM_GAUSS * this->conv_in_channels_ * this->conv_out_channels_;

    for (int i = 0; i < bottom.size(); ++i) {

        // input data
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* bottom_error = bottom[i]->mutable_cpu_diff();

        // actual output data
        const Dtype* top_data = top[i]->cpu_data();
        const Dtype* top_error = top[i]->cpu_diff();

        // Gradient w.r.t. bias.
        if (this->bias_term_ && this->param_propagate_down_[1]) {

            /*CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0],
                                                     cudnn::dataType<Dtype>::one,
                                                     top_bias_descs_[i],  top_error,
                                                     cudnn::dataType<Dtype>::one,
                                                     bias_desc_, bias_diff));*/

        }

        // Gradient w.r.t w,mu1,mu2 and sigma
        if (this->param_propagate_down_[0]) {

            for (int k = 0; k < NUM_K; ++k) {
                printf("k=%d\n",k);
                if (0)
                // perform pre-filtering for each parameter i.e. with four different derivative filters
                CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
                                                    cudnn::dataType<Dtype>::one,
                                                    bottom_descs_[i], bottom_data,
                                                    bwd_g_kernel_desc_, deriv_kernels_data,
                                                    bwd_conv_data_descs_[i],
                                                    bwd_data_algo_[i], workspace[0], workspace_bwd_data_sizes_[i],
                                                    cudnn::dataType<Dtype>::zero,
                                                    bwd_interm_data_descs_[i], interm_data + k * param_size));

                // TODO: update support for K=4 as well
                interm_data = (Dtype*)bottom_data;
                // collect gradients by shifting convolved bottom input data and multiplying it with the top error data

                offset_and_dot_opencv(interm_data, //+ k * param_size,
                                      top_error,
                                      filter_weights, filter_offsets_float_mu1, filter_offsets_float_mu2,
                                      bwd_gradients_data + k * param_size,
                                      this->num_, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_,
                                      this->width_, this->height_,
                                      this->width_out_, this->height_out_, this->ignore_edge_gradients_);

            }
        }

        if (0)
            // finally perform back-propagation of the error values
            if (propagate_down[i]) {
                // we need to do pre-filtering of the error values

                CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
                                                    cudnn::dataType<Dtype>::one,
                                                    top_descs_[i], top_data,
                                                    bwd_g_kernel_desc_, deriv_error_kernel,
                                                    bwd_conv_error_descs_[i],
                                                    bwd_error_algo_[i], workspace[0], workspace_bwd_error_sizes_[i],
                                                    cudnn::dataType<Dtype>::zero,
                                                    bwd_interm_error_descs_[i], interm_data));

                // then use our custom kernel for forwarding, however we need to transpose kernels, which in our case means
                // that we need to rotate mu1,mu2 locations

                // we can re-use bwd_gradients_data buffer for mu1 and mu2 that are rotated
                Dtype *param_mu1_backprop = this->tmp_param_buffer_.mutable_gpu_data() + 0 * param_size;
                Dtype *param_mu2_backprop = this->tmp_param_buffer_.mutable_gpu_data() + 1 * param_size;

                // rot(mu) = (kernel_w-1) - mu
                {
                    caffe_gpu_memcpy(param_size, filter_offsets_float_mu1, param_mu1_backprop);
                    caffe_gpu_memcpy(param_size, filter_offsets_float_mu2, param_mu2_backprop);

                    caffe_gpu_scal(param_size, (Dtype)-1, param_mu1_backprop);
                    caffe_gpu_scal(param_size, (Dtype)-1, param_mu2_backprop);

                    caffe_gpu_add_scalar(param_size, (Dtype)(this->gauss_kernel_w_ - 1), param_mu1_backprop);
                    caffe_gpu_add_scalar(param_size, (Dtype)(this->gauss_kernel_h_ - 1), param_mu2_backprop);
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
    }
}

INSTANTIATE_CLASS(FastAproxGaussianConvLayer);

}   // namespace caffe
#endif
