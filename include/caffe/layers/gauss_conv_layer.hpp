#ifndef CAFFE_GAUSS_CONV_LAYER_HPP_
#define CAFFE_GAUSS_CONV_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

//#include "caffe/vision_layers.hpp"
#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {


template <typename Dtype>
class BaseGaussianConvLayer : public BaseConvolutionLayer<Dtype> {
 public:
	explicit BaseGaussianConvLayer(const LayerParameter& param)
		: BaseConvolutionLayer<Dtype>(param), using_gpu(false) {}

	virtual ~BaseGaussianConvLayer() {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


	void precompute_guassian_weights(bool precompute_derivs);
	void precompute_guassian_weights_gpu(bool precompute_derivs);

	virtual inline bool reverse_dimensions() { return false; }
	virtual void compute_output_shape();



	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int pad_h_, pad_w_;
	int height_, width_;
	int height_out_, width_out_;
	int conv_in_height_;
	int conv_in_width_;
	int weight_offset_;

	bool use_gmm_weight_normalization;
	bool use_gmm_gauss_normalization;
	bool use_gmm_square_gauss_normalization;
	bool use_gmm_seperable_kernels;

	Dtype gmm_component_border_bound;
	Dtype gmm_sigma_lower_bound;

    int gmm_mean_iteration_step;
	int gmm_sigma_iteration_step;
	int gmm_merge_iteration_step;
	float gmm_merge_threshold;

	bool gmm_discretize_mean;

	int NUM_GAUSS;

	// parameters to learn
	shared_ptr<Blob<Dtype> > param_buffer_w_;
	shared_ptr<Blob<Dtype> > param_buffer_mu1_;
	shared_ptr<Blob<Dtype> > param_buffer_mu2_;
	shared_ptr<Blob<Dtype> > param_buffer_sigma_;
	shared_ptr<Blob<Dtype> > param_buffer_bias_;

	// temporary buffers for discretized mean parameters
	shared_ptr<Blob<Dtype> > param_buffer_mu1_discr_;
	shared_ptr<Blob<Dtype> > param_buffer_mu2_discr_;

	// temporary buffers for pre-computed sigma^2, sigma^3 and 1/2*sigma^2
	Blob<Dtype> param_buffer_sigma_square_inv_;
	Blob<Dtype> param_buffer_sigma_cube_inv_;
	Blob<Dtype> param_buffer_sigma_square_inv_half_;

	// weight kernels computed from Guassian component parameters
	shared_ptr<Blob<Dtype> > weight_buffer_;
	shared_ptr<Blob<Dtype> > deriv_error_buffer_;
	shared_ptr<Blob<Dtype> > deriv_weight_buffer_;
	shared_ptr<Blob<Dtype> > deriv_sigma_buffer_;
	shared_ptr<Blob<Dtype> > deriv_mu1_buffer_;
	shared_ptr<Blob<Dtype> > deriv_mu2_buffer_;

	// boolean buffer for flagging enabled/disabled Guassian components
	Blob<int> is_weight_enabled_buffer_;

	Blob<int> tmp_precomp_index_; // pre-computed indexes for caffe_gpu_sum in precompute_guassian_weights_gpu
	int* tmp_precomp_index_gpu;

	Blob<Dtype> tmp_deriv_weight_buffer_; // temporary buffer for holding kernel weights when calling precompute_guassian_weights

	// intermediate buffers when computing derivative kernels in precompute_guassian_weights_gpu
	Blob<Dtype> guass_dist_buffer_;
	Blob<Dtype> gauss_dist_square_buffer_;
	Blob<Dtype> deriv_mu1_times_gauss_dist_buffer_;
	Blob<Dtype> deriv_mu2_times_gauss_dist_buffer_;
	Blob<Dtype> deriv_sigma_times_gauss_dist_buffer_;

	Blob<Dtype> guass_norm_buffer_;
	Blob<Dtype> deriv_mu1_sums_buffer_;
	Blob<Dtype> deriv_mu2_sums_buffer_;
	Blob<Dtype> deriv_sigma_sums_buffer_;

	// separable kernels
	shared_ptr<Blob<Dtype> > weight_vert_buffer_; // vertical weights
	shared_ptr<Blob<Dtype> > weight_horiz_buffer_; // Horizontal weights

	shared_ptr<Blob<Dtype> > random_mu1_buffer_;
	shared_ptr<Blob<Dtype> > random_mu2_buffer_;

	bool using_gpu;

    int current_iteration_index;

    int num_extra_blobs;
};

template <typename Dtype>
class GaussianConvLayer : public BaseGaussianConvLayer<Dtype> {
 public:
  
  explicit GaussianConvLayer(const LayerParameter& param)
      : BaseGaussianConvLayer<Dtype>(param),  A(0), B(0), C(0), d_A(0), d_B(0), d_C(0) {}

  virtual ~GaussianConvLayer();

  virtual inline const char* type() const { return "GaussianConv"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
 //protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void compute_parameter_deriv(int num_iter,const Blob<Dtype>& col_activations_buffer, const Blob<Dtype>& deriv_kernels_buffer, 
				//const Blob<Dtype>& top_error_buffer, Blob<Dtype>& param_output_buffer, int param_output_offset);
		  	  	  Blob<Dtype>& top_error_buffer, Blob<Dtype>& param_output_buffer, int param_output_offset);

  // temporary accumulation buffer for testing GPU performance in backward-pass
  Blob<Dtype> accum_bottom_;

  Blob<Dtype> tmp_blob_; // used by backward-pass in compute_parameter_deriv
  Blob<Dtype> tmp_buffer_; // used by backward-pass in compute_parameter_deriv

  Blob<int> tmp_index_; // pre-computed indexes for caffe_gpu_sum in compute_parameter_deriv
  int* tmp_index_gpu;

  Blob<Dtype> tmp_w_sign_; // tmp buffer for sign when use_gmm_weight_normalization==1
  Blob<Dtype> tmp_w_fabs_; // tmp buffer for fabs when use_gmm_weight_normalization==1

  // buffers for forward-pass with separable kernels
  Blob<Dtype> tmp_buffer_sepe_1_;
  Blob<Dtype> tmp_buffer_sepe_2_;

//  bool using_gpu;


  // temporary buffer for holding pointers to data in compute_parameter_deriv when doing batched gemm
  const  Dtype** A;
  const Dtype** B;
  Dtype** C;

  Dtype **d_A, **d_B, **d_C;

  void forward_cpu_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col = false);
  void forward_cpu_gpu_seperable(const Dtype* input, const Dtype* weights_vert, const Dtype* weights_horiz, const int* is_weight_enabled, Dtype* output, Dtype* col_buff, Dtype* second_col_buff);
  void forward_cpu_gpu_bias(Dtype* output, const Dtype* bias);

  void weight_cpu_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);

  void backward_cpu_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input);
  void backward_cpu_gpu_bias(Dtype* bias, const Dtype* input);

  void caffe_cpu_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C);
  void caffe_cpu_gpu_gemm_batched(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const Dtype alpha, const Dtype** A, const Dtype** B, const Dtype beta, Dtype** C, int batch_count);

};

template <typename Dtype>
class FastAproxGaussianConvLayer : public BaseGaussianConvLayer<Dtype> {
 public:

  explicit FastAproxGaussianConvLayer(const LayerParameter& param)
      : BaseGaussianConvLayer<Dtype>(param){}

  virtual ~FastAproxGaussianConvLayer();

  virtual inline const char* type() const { return "FastAproxGaussianConvLayer"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

 //protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void test_kernel_cpu(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, float* output, const int I, const int S, const int F, const int G, const int img_width, const int img_height, const int kernel_width, const int kernel_height, const bool use_interpolation);
  void test_kernel_gpu(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, float* output, const int I, const int S, const int F, const int G, const int img_width, const int img_height, const int kernel_width, const int kernel_height, const bool use_interpolation);
  void test_backward_kernel_gpu(const float* filtered_images, const float* error_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, float* output, const int I, const int S, const int F, const int G, const int img_width, const int img_height, const int kernel_width, const int kernel_height) ;
  void test_backward_multi_subfeature_kernel_gpu(const float* filtered_images, const float* error_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, float* output, const int K, const int I, const int S, const int F, const int G, const int img_width, const int img_height, const int kernel_width, const int kernel_height, const bool use_interpolation);



};


#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNGaussianConvLayer : public BaseGaussianConvLayer<Dtype> {
 public:
  explicit CuDNNGaussianConvLayer(const LayerParameter& param)
	  : BaseGaussianConvLayer<Dtype>(param), handles_setup_(false), dirty_gauss_dist_buffer(true), dirty_norm_buffer(true), dirty_norm_with_w_buffer(true), dirty_weight_buffer(true), dirty_weight_deriv_buffer(true), dirty_mu1_deriv_buffer(true), dirty_mu2_deriv_buffer(true), dirty_sigma_deriv_buffer(true) {}

  virtual ~CuDNNGaussianConvLayer();

  virtual inline const char* type() const { return "CuDNNGaussianConv"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  shared_ptr<Blob<Dtype> > get_gauss_distribution_buffer(cudaStream_t streamId = 0);
  shared_ptr<Blob<Dtype> > get_gauss_normalization_with_weight_buffer(cudaStream_t streamId = 0);
  shared_ptr<Blob<Dtype> > get_gauss_normalization_buffer(cudaStream_t streamId = 0);

  shared_ptr<Blob<Dtype> > get_weight_filters(cudaStream_t streamId = 0);
  shared_ptr<Blob<Dtype> > get_weight_derivative_filters(shared_ptr<Blob<Dtype> > output_buffer, cudaStream_t streamId = 0);
  shared_ptr<Blob<Dtype> > get_mu1_derivative_filters(shared_ptr<Blob<Dtype> > output_buffer, shared_ptr<Blob<Dtype> > deriv_weight_buffer, cudaStream_t streamId = 0);
  shared_ptr<Blob<Dtype> > get_mu2_derivative_filters(shared_ptr<Blob<Dtype> > output_buffer, shared_ptr<Blob<Dtype> > deriv_weight_buffer, cudaStream_t streamId = 0);
  shared_ptr<Blob<Dtype> > get_sigma_derivative_filters(shared_ptr<Blob<Dtype> > output_buffer, shared_ptr<Blob<Dtype> > deriv_weight_buffer, cudaStream_t streamId = 0);

  void merge_components(Blob<Dtype>* scores = NULL);

  void fill_random_mean(Blob<Dtype>* random_mu1, Blob<Dtype>* random_mu2);

  void set_buffers_dirty() {
	  dirty_gauss_dist_buffer = true;
	  dirty_norm_buffer = true;
	  dirty_norm_with_w_buffer = true;

	  dirty_weight_buffer = true;
	  dirty_weight_deriv_buffer = true;
	  dirty_mu1_deriv_buffer = true;
	  dirty_mu2_deriv_buffer = true;
	  dirty_sigma_deriv_buffer = true;
  }

  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t*  stream_;

  cudaStream_t* paralel_streams; // parallel streams for custom back-propagation kernels

  // algos and descriptors for forward convolution
  cudnnConvolutionFwdAlgo_t *fwd_algo_;

  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;

  // algos and descriptors for backward convolution
  cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData

  int num_guass_per_compute;

  /// pre-compute variables

  bool dirty_gauss_dist_buffer;
  bool dirty_norm_buffer;
  bool dirty_norm_with_w_buffer;

  bool dirty_weight_buffer;
  bool dirty_weight_deriv_buffer;
  bool dirty_mu1_deriv_buffer;
  bool dirty_mu2_deriv_buffer;
  bool dirty_sigma_deriv_buffer;

  // weight kernels computed from Guassian component parameters
  //shared_ptr<Blob<Dtype> > weight_buffer_;
  // two deriv buffers of size [S * G * F * K_h * K_w]
  struct {
	  shared_ptr<Blob<Dtype> > weights; // used for weight kernel (as well as, deriv of mu1, deriv of mu2 and deriv of sigma kernels when in release mode)
	  shared_ptr<Blob<Dtype> > deriv_weight; // used for deriv of weights
	  shared_ptr<Blob<Dtype> > deriv_mu1; // used for deriv of mu1
	  shared_ptr<Blob<Dtype> > deriv_mu2; // used for deriv of mu2
	  shared_ptr<Blob<Dtype> > deriv_sigma; // used for deriv of sigma

  } kernel_buf;

  // intermediate buffers when computing derivative kernels in precompute_guassian_weights_gpu
  struct {
	  shared_ptr<Blob<Dtype> > distribution;	// [S * G * F * K_h * K_w]

	  // buffers for pre-computed sigma^2, sigma^3 and 1/2*sigma^2 (they have small memory requirements - size [S x F x G])
	  shared_ptr<Blob<Dtype> > sigma_square_inv;
	  shared_ptr<Blob<Dtype> > sigma_cube_inv;
	  shared_ptr<Blob<Dtype> > sigma_square_inv_half;

	  // normalization buffers are small - only [S x F x G] in size
	  shared_ptr<Blob<Dtype> > norm;
	  shared_ptr<Blob<Dtype> > norm_with_w;

	  // temporary sum buffers (they may not be needed but are only of [S x F x G] size - should be realy small)
	  shared_ptr<Blob<Dtype> > deriv_mu1_sums;
	  shared_ptr<Blob<Dtype> > deriv_mu2_sums;
	  shared_ptr<Blob<Dtype> > deriv_sigma_sums;

	  // buffers for holding random variables during mering of components - these buffers MUST be part of this->blobs_ so that they are sync among multiple-gpus
	  shared_ptr<Blob<Dtype> > random_mu1;
	  shared_ptr<Blob<Dtype> > random_mu2;
  } tmp_buf;

};

template <typename Dtype>
class CuDNNOldGaussianConvLayer : public CuDNNGaussianConvLayer<Dtype> {
 public:
  explicit CuDNNOldGaussianConvLayer(const LayerParameter& param)
	  : CuDNNGaussianConvLayer<Dtype>(param), tmp_buffer_1_gpu(NULL) {}

  virtual ~CuDNNOldGaussianConvLayer();

  virtual inline const char* type() const { return "CuDNNOldGaussianConvLayer"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void compute_parameter_deriv(const int sample_index,
						  	   const int group_index,
							   const int subfeature_index,
							   const Dtype* bottom_data,
							   const Dtype* top_diff, const int top_count,
							   const Blob<Dtype>* deriv_kernel, const Dtype* deriv_kernel_data,
							   Blob<Dtype>* param_buffer, Dtype* param_buffer_diff,
							   Dtype* intermediate_buff, Dtype* intermediate_sum_buff, int* intermediate_sum_index, int * top_remapping_index, int stream_offset);

   // algos and descriptors for backward convolution
   cudnnConvolutionFwdAlgo_t *bwd_filter_algo_;

   vector<cudnnTensorDescriptor_t> backward_bottom_desc_, backward_intermed_desc_;
   cudnnFilterDescriptor_t      backward_filter_desc_;

   size_t *workspace_bwd_filter_sizes_;

  // buffer for indexes in top when doing back-propagation in compute_parameter_deriv using standard cuBlas function (requires extra memory)
  Blob<int> tmp_buffer_1;
  int* tmp_buffer_1_gpu;

  Blob<Dtype> tmp_blob_;
  Blob<Dtype> tmp_buffer_;

  Blob<int> tmp_index_;
  int* tmp_index_gpu;

};

#endif

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
