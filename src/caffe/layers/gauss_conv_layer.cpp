#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//#define NUM_GAUSS 4
#define NUM_GAUSS_COMPONENT_PARAM 4
#define NUM_GAUSS_PARAM 1

namespace caffe {


template <typename Dtype>
void GaussianConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	int NUM_GAUSS_PER_AXIS = this->layer_param_.convolution_param().number_gauss();
	int NUM_GAUSS =  NUM_GAUSS_PER_AXIS * NUM_GAUSS_PER_AXIS;

	// TODO: with new changes in master kernel_size, pad and stripe are repeated fields
	//       and this code needs to be changes to use that
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
	// Configure the kernel size, padding, stride, and inputs.
	ConvolutionParameter conv_param = this->layer_param_.convolution_param();
	CHECK(!conv_param.kernel_size_size() !=
			!(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
	<< "Filter size is kernel_size_size OR kernel_h and kernel_w; not both";
	CHECK(conv_param.kernel_size_size() ||
			(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
	<< "For non-square filters both kernel_h and kernel_w are required.";
	CHECK((!conv_param.pad_size() && conv_param.has_pad_h()
			&& conv_param.has_pad_w())
			|| (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
	<< "pad is pad OR pad_h and pad_w are required.";
	CHECK((!conv_param.stride_size() && conv_param.has_stride_h()
			&& conv_param.has_stride_w())
			|| (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
	<< "Stride is stride OR stride_h and stride_w are required.";
	if (conv_param.kernel_size_size()) {
		this->kernel_h_ = this->kernel_w_ = conv_param.kernel_size(0);
	} else {
		this->kernel_h_ = conv_param.kernel_h();
		this->kernel_w_ = conv_param.kernel_w();
	}
	CHECK_GT(this->kernel_h_, 0) << "Filter dimensions cannot be zero.";
	CHECK_GT(this->kernel_w_, 0) << "Filter dimensions cannot be zero.";
	if (!conv_param.has_pad_h()) {
		this->pad_h_ = this->pad_w_ = conv_param.pad(0);
	} else {
		this->pad_h_ = conv_param.pad_h();
		this->pad_w_ = conv_param.pad_w();
	}
	if (!conv_param.has_stride_h()) {
		this->stride_h_ = this->stride_w_ = conv_param.stride(0);
	} else {
		this->stride_h_ = conv_param.stride_h();
		this->stride_w_ = conv_param.stride_w();
	}

	this->force_nd_im2col_ = conv_param.force_nd_im2col();
	this->channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
	const int first_spatial_axis = this->channel_axis_ + 1;
	const int num_axes = bottom[0]->num_axes();
	this->num_spatial_axes_ = num_axes - first_spatial_axis;
	CHECK_GE(this->num_spatial_axes_, 0);
	vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
	vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));
	// Setup filter kernel dimensions (kernel_shape_).
	this->kernel_shape_.Reshape(spatial_dim_blob_shape);
	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
		CHECK_EQ(this->num_spatial_axes_, 2)
			<< "kernel_h & kernel_w can only be used for 2D convolution.";
		CHECK_EQ(0, conv_param.kernel_size_size())
			<< "Either kernel_size or kernel_h/w should be specified; not both.";
		kernel_shape_data[0] = conv_param.kernel_h();
		kernel_shape_data[1] = conv_param.kernel_w();
	} else {
		const int num_kernel_dims = conv_param.kernel_size_size();
		CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
			<< "kernel_size must be specified once, or once per spatial dimension "
			<< "(kernel_size specified " << num_kernel_dims << " times; "
			<< this->num_spatial_axes_ << " spatial dims);";
		for (int i = 0; i < this->num_spatial_axes_; ++i) {
			kernel_shape_data[i] = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
		}
	}
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
	}
	// Setup stride dimensions (stride_).
	this->stride_.Reshape(spatial_dim_blob_shape);
	int* stride_data = this->stride_.mutable_cpu_data();
	if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
		CHECK_EQ(this->num_spatial_axes_, 2)
			<< "stride_h & stride_w can only be used for 2D convolution.";
		CHECK_EQ(0, conv_param.stride_size())
			<< "Either stride or stride_h/w should be specified; not both.";
		stride_data[0] = conv_param.stride_h();
		stride_data[1] = conv_param.stride_w();
	} else {
		const int num_stride_dims = conv_param.stride_size();
		CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
			  num_stride_dims == this->num_spatial_axes_)
			<< "stride must be specified once, or once per spatial dimension "
			<< "(stride specified " << num_stride_dims << " times; "
			<< this->num_spatial_axes_ << " spatial dims);";
		const int kDefaultStride = 1;
		for (int i = 0; i < this->num_spatial_axes_; ++i) {
		  stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
			  conv_param.stride((num_stride_dims == 1) ? 0 : i);
		  CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
		}
	}
	// Setup pad dimensions (pad_).
	this->pad_.Reshape(spatial_dim_blob_shape);
	int* pad_data = this->pad_.mutable_cpu_data();
	if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
		CHECK_EQ(this->num_spatial_axes_, 2)
			<< "pad_h & pad_w can only be used for 2D convolution.";
		CHECK_EQ(0, conv_param.pad_size())
			<< "Either pad or pad_h/w should be specified; not both.";
		pad_data[0] = conv_param.pad_h();
		pad_data[1] = conv_param.pad_w();
	} else {
		const int num_pad_dims = conv_param.pad_size();
		CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
			  num_pad_dims == this->num_spatial_axes_)
			<< "pad must be specified once, or once per spatial dimension "
			<< "(pad specified " << num_pad_dims << " times; "
			<< this->num_spatial_axes_ << " spatial dims);";
		const int kDefaultPad = 0;
		for (int i = 0; i < this->num_spatial_axes_; ++i) {
		  pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
			  conv_param.pad((num_pad_dims == 1) ? 0 : i);
		}
	}

	//CHECK_EQ(this->stride_h_, 1) << "Only stride of 1 is supported";
	//CHECK_EQ(this->stride_w_, 1) << "Only stride of 1 is supported";

	// Special case: im2col is the identity for 1x1 convolution with stride 1
	// and no padding, so flag for skipping the buffer and transformation.
	this->is_1x1_ = this->kernel_w_ == 1 && this->kernel_h_ == 1
			&& this->stride_h_ == 1 && this->stride_w_ == 1 && this->pad_h_ == 0 && this->pad_w_ == 0;
	this->force_nd_im2col_ = false;
	// Configure output channels and groups.
	this->channels_ = bottom[0]->channels();
	this->num_output_ = this->layer_param_.convolution_param().num_output();
	CHECK_GT(this->num_output_, 0);
	this->group_ = this->layer_param_.convolution_param().group();
	CHECK_EQ(this->channels_ % this->group_, 0);
	CHECK_EQ(this->num_output_ % this->group_, 0)
	<< "Number of output should be multiples of group.";
	if (reverse_dimensions()) {
		this->conv_out_channels_ = this->channels_;
		this->conv_in_channels_ = this->num_output_;
	} else {
		this->conv_out_channels_ = this->num_output_;
		this->conv_in_channels_ = this->channels_;
	}

	// Handle the parameters: weights and biases.
	// - blobs_[0] holds the filter weights
	// - blobs_[1] holds the biases (optional)
	this->bias_term_ = this->layer_param_.convolution_param().bias_term();
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		if (this->bias_term_) {
			this->blobs_.resize(1 + NUM_GAUSS_COMPONENT_PARAM + NUM_GAUSS_PARAM + 6 );
			//this->blobs_.resize(1 + NUM_GAUSS_COMPONENT_PARAM + NUM_GAUSS_PARAM);
		} else {
			this->blobs_.resize(NUM_GAUSS_COMPONENT_PARAM +  NUM_GAUSS_PARAM + 6 );
			//this->blobs_.resize(NUM_GAUSS_COMPONENT_PARAM +  NUM_GAUSS_PARAM);
		}
		// Initialize and fill the weights:
		// output channels x input channels per-group x kernel height x kernel width
		// set size of all per-components GMM paramteres (weight,mu1,mu2,sigma)
		int blobs_index = 0;
		for (int i = 0; i < NUM_GAUSS_COMPONENT_PARAM; i++)
			this->blobs_[blobs_index++].reset(new Blob<Dtype>(1, this->conv_in_channels_, this->conv_out_channels_, NUM_GAUSS));

		// set per-GMM parameters (weight_gmm) - depricated, NUM_GAUSS_PARAM should be 0
		for (int i = 0; i < NUM_GAUSS_PARAM; i++)
			this->blobs_[blobs_index++].reset(new Blob<Dtype>(1, 1, this->conv_out_channels_, this->conv_in_channels_));

		// If necessary, initialize the biases.
		if (this->bias_term_) {
			vector<int> bias_shape(1, this->num_output_);
			this->blobs_[blobs_index++].reset(new Blob<Dtype>(bias_shape));
		}

		this->param_buffer_w_ = this->blobs_[0];
		this->param_buffer_mu1_ = this->blobs_[1];
		this->param_buffer_mu2_ = this->blobs_[2];
		this->param_buffer_sigma_ = this->blobs_[3];
		this->param_buffer_bias_ = this->blobs_[5];

		Blob<Dtype> tmp_w(1, this->conv_in_channels_, this->conv_out_channels_, NUM_GAUSS);
		Blob<Dtype> tmp_sigma(1, this->conv_in_channels_, this->conv_out_channels_, NUM_GAUSS);
		Blob<Dtype> tmp_mu1(1, this->conv_in_channels_, this->conv_out_channels_, NUM_GAUSS);
		Blob<Dtype> tmp_mu2(1, this->conv_in_channels_, this->conv_out_channels_, NUM_GAUSS);

		shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.convolution_param().weight_filler()));
		shared_ptr<Filler<Dtype> > sigma_filler(GetFiller<Dtype>(this->layer_param_.convolution_param().sigma_filler()));
		shared_ptr<Filler<Dtype> > mu_filler(GetFiller<Dtype>(this->layer_param_.convolution_param().mu_filler()));

		weight_filler->Fill(&tmp_w);
		sigma_filler->Fill(&tmp_sigma);
		//mu_filler->Fill(&tmp_mu1);
		//mu_filler->Fill(&tmp_mu2);

		Dtype* mu1_buf = tmp_mu1.mutable_cpu_data();
		Dtype* mu2_buf = tmp_mu2.mutable_cpu_data();

		//int num_gauss_per_axis = NUM_GAUSS /2;
		int* offset_x = new int[NUM_GAUSS_PER_AXIS];
		int* offset_y = new int[NUM_GAUSS_PER_AXIS];

		for (int i = 0; i < NUM_GAUSS_PER_AXIS; i++) {
			offset_x[i] = (i+1)*this->kernel_w_ /(NUM_GAUSS_PER_AXIS+1);
			offset_y[i] = (i+1)*this->kernel_h_ /(NUM_GAUSS_PER_AXIS+1);
		}

		for (int i = 0; i < this->conv_in_channels_*this->conv_out_channels_; ++i) {
			for (int j = 0; j < NUM_GAUSS; j++) {
				mu1_buf[i * NUM_GAUSS + j] = offset_x[j / NUM_GAUSS_PER_AXIS];
				mu2_buf[i * NUM_GAUSS + j] = offset_y[j %  NUM_GAUSS_PER_AXIS];
			}
		}
		delete [] offset_x;
		delete [] offset_y;

		memcpy(this->param_buffer_w_->mutable_cpu_data() + this->param_buffer_w_->offset(0), tmp_w.cpu_data(), sizeof(Dtype) * this->conv_in_channels_ * this->conv_out_channels_ * NUM_GAUSS);
		memcpy(this->param_buffer_mu1_->mutable_cpu_data() + this->param_buffer_mu1_->offset(0), tmp_mu1.cpu_data(), sizeof(Dtype) * this->conv_in_channels_ * this->conv_out_channels_ * NUM_GAUSS);
		memcpy(this->param_buffer_mu2_->mutable_cpu_data() + this->param_buffer_mu2_->offset(0), tmp_mu2.cpu_data(), sizeof(Dtype) * this->conv_in_channels_ * this->conv_out_channels_ * NUM_GAUSS);
		memcpy(this->param_buffer_sigma_->mutable_cpu_data() + this->param_buffer_sigma_->offset(0), tmp_sigma.cpu_data(), sizeof(Dtype) * this->conv_in_channels_ * this->conv_out_channels_ * NUM_GAUSS);

		// If necessary, fill the biases.
		if (this->bias_term_) {
			shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(this->layer_param_.convolution_param().bias_filler()));
			bias_filler->Fill(this->param_buffer_bias_.get());
		}

		int precomputed_blobs_offset = NUM_GAUSS_COMPONENT_PARAM + NUM_GAUSS_PARAM;
		if (this->bias_term_)
			precomputed_blobs_offset++;

		weight_buffer_.reset(new Blob<Dtype>());
		deriv_error_buffer_.reset(new Blob<Dtype>());
		deriv_weight_buffer_.reset(new Blob<Dtype>());
		deriv_sigma_buffer_.reset(new Blob<Dtype>());
		deriv_mu1_buffer_.reset(new Blob<Dtype>());
		deriv_mu2_buffer_.reset(new Blob<Dtype>());

		this->blobs_[precomputed_blobs_offset + 0] = this->weight_buffer_;
		this->blobs_[precomputed_blobs_offset + 1] = this->deriv_error_buffer_;
		this->blobs_[precomputed_blobs_offset + 2] = this->deriv_weight_buffer_;
		this->blobs_[precomputed_blobs_offset + 3] = this->deriv_sigma_buffer_;
		this->blobs_[precomputed_blobs_offset + 4] = this->deriv_mu1_buffer_;
		this->blobs_[precomputed_blobs_offset + 5] = this->deriv_mu2_buffer_;
	}
	// Propagate gradients to the parameters (as directed by backward pass).
	this->param_propagate_down_.resize(this->blobs_.size(), true);

	// decide if needed to perform gmm weight normalization
	this->use_gmm_weight_normalization = this->layer_param_.convolution_param().gmm_weight_normalization();
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const int first_spatial_axis = this->channel_axis_ + 1;
	int NUM_GAUSS_PER_AXIS = this->layer_param_.convolution_param().number_gauss();
	int NUM_GAUSS =  NUM_GAUSS_PER_AXIS * NUM_GAUSS_PER_AXIS;

	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
	this->num_ = bottom[0]->num();
	this->height_ = bottom[0]->height();
	this->width_ = bottom[0]->width();
	CHECK_EQ(bottom[0]->channels(), this->channels_) << "Input size incompatible with"
			" convolution kernel.";
	// TODO: generalize to handle inputs of different shapes.
	for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
		CHECK_EQ(this->num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
		CHECK_EQ(this->channels_, bottom[bottom_id]->channels())
		<< "Inputs must have same channels.";
		CHECK_EQ(this->height_, bottom[bottom_id]->height())
		<< "Inputs must have same height.";
		CHECK_EQ(this->width_, bottom[bottom_id]->width())
		<< "Inputs must have same width.";
	}
	// Shape the tops.
	compute_output_shape();
	for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->Reshape(this->num_, this->num_output_, this->height_out_, this->width_out_);
	}
	if (reverse_dimensions()) {
		this->conv_in_height_ = this->height_out_;
		this->conv_in_width_ = this->width_out_;
		this->conv_out_spatial_dim_ = this->height_ * this->width_;
	} else {
		this->conv_in_height_ = this->height_;
		this->conv_in_width_ = this->width_;
		this->conv_out_spatial_dim_ = this->height_out_ * this->width_out_;
	}

	this->out_spatial_dim_ = top[0]->count(first_spatial_axis);
	this->bottom_dim_ = bottom[0]->count(this->channel_axis_);
	this->top_dim_ = top[0]->count(this->channel_axis_);
	this->num_kernels_im2col_ = this->conv_in_channels_ * this->conv_out_spatial_dim_;
	this->num_kernels_col2im_ = reverse_dimensions() ? this->top_dim_ : this->bottom_dim_;

	this->kernel_dim_ = this->conv_in_channels_ * this->kernel_h_ * this->kernel_w_;
	this->weight_offset_ = this->conv_out_channels_ * this->kernel_dim_ / this->group_ / this->group_;
	this->col_offset_ = this->kernel_dim_ * this->conv_out_spatial_dim_ / this->group_;
	this->output_offset_ = this->conv_out_channels_ * this->conv_out_spatial_dim_ / this->group_;

	 // Setup input dimensions (conv_input_shape_).
	vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
	this->conv_input_shape_.Reshape(bottom_dim_blob_shape);
	int* conv_input_shape_data = this->conv_input_shape_.mutable_cpu_data();
	for (int i = 0; i < this->num_spatial_axes_ + 1; ++i) {
		if (reverse_dimensions()) {
			conv_input_shape_data[i] = top[0]->shape(this->channel_axis_ + i);
		} else {
			conv_input_shape_data[i] = bottom[0]->shape(this->channel_axis_ + i);
		}
	}

	// The im2col result buffer will only hold one image at a time to avoid
	// overly large memory usage. In the special case of 1x1 convolution
	// it goes lazily unused to save memory.
	if (reverse_dimensions()) {
		this->col_buffer_.Reshape(this->conv_in_channels_, this->kernel_h_ * this->kernel_w_, this->height_, this->width_);
	} else {
		this->col_buffer_.Reshape(this->conv_in_channels_, this->kernel_h_ * this->kernel_w_, this->height_out_, this->width_out_);
	}

	this->weight_buffer_->Reshape(this->conv_out_channels_, this->conv_in_channels_, this->kernel_h_, this->kernel_w_);
	this->deriv_error_buffer_->Reshape(this->conv_in_channels_, this->conv_out_channels_, this->kernel_h_, this->kernel_w_);
	this->deriv_weight_buffer_->Reshape(this->conv_in_channels_, NUM_GAUSS * this->conv_out_channels_, this->kernel_h_, this->kernel_w_);
	this->deriv_sigma_buffer_->Reshape(this->conv_in_channels_, NUM_GAUSS * this->conv_out_channels_, this->kernel_h_, this->kernel_w_);
	this->deriv_mu1_buffer_->Reshape(this->conv_in_channels_, NUM_GAUSS * this->conv_out_channels_, this->kernel_h_, this->kernel_w_);
	this->deriv_mu2_buffer_->Reshape(this->conv_in_channels_, NUM_GAUSS * this->conv_out_channels_, this->kernel_h_, this->kernel_w_);

	this->tmp_buffer_.Reshape(this->conv_out_channels_, NUM_GAUSS, this->height_out_, this->width_out_);

	this->tmp_deriv_weight_buffer_.Reshape(1, NUM_GAUSS , this->kernel_h_, this->kernel_w_);

	//this->tmp_bottom_buffer_.Reshape(this->conv_out_channels_, this->conv_in_channels_, this->height_out_, this->width_out_);
	this->tmp_bottom_buffer_.Reshape(this->num_, this->conv_in_channels_, this->height_out_, this->width_out_); // for w_gmm without normalization

	this->tmp_w_sign_.ReshapeLike(*this->param_buffer_w_);
	this->tmp_w_fabs_.ReshapeLike(*this->param_buffer_w_);

	// Set up the all ones "bias multiplier" for adding biases by BLAS
	if (this->bias_term_) {
		vector<int> bias_multiplier_shape(1, this->height_out_ * this->width_out_);
		this->bias_multiplier_.Reshape(bias_multiplier_shape);
		caffe_set(this->bias_multiplier_.count(), Dtype(1), this->bias_multiplier_.mutable_cpu_data());
	}
}  

template <typename Dtype>
void GaussianConvLayer<Dtype>::compute_output_shape() {
	this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
    		  / this->stride_h_ + 1;
	this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
    		  / this->stride_w_ + 1;
}

float caffe_sum(const int N, const float* X) {
	float sum = 0;
	for (int i = 0; i < N; ++i) sum += X[i];
	return sum;
}

double caffe_sum(const int N, const double* X) {
	double sum = 0;
	for (int i = 0; i < N; ++i) sum += X[i];
	return sum;
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::precompute_guassian_weights(bool is_backward_pass) {

	int NUM_GAUSS_PER_AXIS = this->layer_param_.convolution_param().number_gauss();
	int NUM_GAUSS =  NUM_GAUSS_PER_AXIS * NUM_GAUSS_PER_AXIS;

	Dtype* weight = this->weight_buffer_->mutable_cpu_data();

	Dtype* deriv_error, *deriv_weight, *deriv_sigma, *deriv_mu1, *deriv_mu2, *tmp_deriv_weight;

	if (is_backward_pass) {
		deriv_error = this->deriv_error_buffer_->mutable_cpu_data();
		deriv_weight = this->deriv_weight_buffer_->mutable_cpu_data();
		deriv_mu1 = this->deriv_mu1_buffer_->mutable_cpu_data();
		deriv_mu2 = this->deriv_mu2_buffer_->mutable_cpu_data();
		deriv_sigma = this->deriv_sigma_buffer_->mutable_cpu_data();

		//tmp_deriv_weight = this->tmp_deriv_weight_buffer_.mutable_cpu_data();
		tmp_deriv_weight = deriv_weight;

		memset(deriv_error, 0, sizeof(Dtype) * this->deriv_error_buffer_->count());
		memset(deriv_weight, 0, sizeof(Dtype) * this->deriv_weight_buffer_->count());
		memset(deriv_mu1, 0, sizeof(Dtype) * this->deriv_mu1_buffer_->count());
		memset(deriv_mu2, 0, sizeof(Dtype) * this->deriv_mu2_buffer_->count());
		memset(deriv_sigma, 0, sizeof(Dtype) * this->deriv_sigma_buffer_->count());

		memset(tmp_deriv_weight, 0, sizeof(Dtype) * this->tmp_deriv_weight_buffer_.count());
	}

	memset(weight, 0, sizeof(Dtype) * this->weight_buffer_->count());

	// pre-compute weights from guassian params
	Blob<Dtype>& gauss_param_buffer_w = *this->param_buffer_w_;
	Blob<Dtype>& gauss_param_buffer_mu1 = *this->param_buffer_mu1_;
	Blob<Dtype>& gauss_param_buffer_mu2 = *this->param_buffer_mu2_;
	Blob<Dtype>& gauss_param_buffer_sigma = *this->param_buffer_sigma_;

	const Dtype* gauss_params_w = gauss_param_buffer_w.cpu_data();
	const Dtype* gauss_params_mu1 = gauss_param_buffer_mu1.cpu_data();
	const Dtype* gauss_params_mu2 = gauss_param_buffer_mu2.cpu_data();
	const Dtype* gauss_params_sigma = gauss_param_buffer_sigma.cpu_data();

	int kernel_size = this->kernel_h_ * this->kernel_w_;

	Dtype* weight_tmp = (Dtype*)malloc(sizeof(Dtype)*kernel_size);

	for (int i = 0; i < this->conv_out_channels_; ++i) { // over each feature (i.e out feature)

		// new weight normalization over all subfeatures
		Dtype w_sum = 0;
		if (this->use_gmm_weight_normalization) {
			for (int j = 0; j < this->conv_in_channels_; ++j) // sum weigths over all subfeature of feature f
				w_sum += caffe_cpu_asum(NUM_GAUSS, gauss_params_w + gauss_param_buffer_w.offset(0,j,i));
		} else {
			// set sum to 1 if we do not do weight normalization
			w_sum = 1;
		}

		Dtype w_sum_2 = w_sum*w_sum;

		for (int j = 0; j < this->conv_in_channels_; ++j) { // over each subfeature (i.e. in feature)

			int weights_offset = this->weight_buffer_->offset(i,j);

			// original weight normalization within subfeature only
			//Dtype w_sum = caffe_cpu_asum(NUM_GAUSS, gauss_params_w + gauss_param_buffer_w.offset(0,j,i));
			//Dtype w_sum_2 = w_sum*w_sum;

			for (int k = 0; k < NUM_GAUSS; ++k) {
				Dtype w = 	gauss_params_w[gauss_param_buffer_w.offset(0,j,i,k)];
				Dtype mu1 = 	gauss_params_mu1[gauss_param_buffer_mu1.offset(0,j,i,k)];
				Dtype mu2 = 	gauss_params_mu2[gauss_param_buffer_mu2.offset(0,j,i,k)];
				Dtype sigma = 	gauss_params_sigma[gauss_param_buffer_sigma.offset(0,j,i,k)];

				Dtype w_org = w;
				w = w/w_sum;

				int deriv_error_offset,deriv_weight_offset,deriv_sigma_offset,deriv_mu1_offset,deriv_mu2_offset, tmp_deriv_weight_offset;

				if (is_backward_pass) {
					// notice: deriv_error_buffer_ has first two dimensions switched to enable more efficent computation of derivates in backward process
					deriv_weight_offset = this->deriv_weight_buffer_->offset(j,i * NUM_GAUSS + k);
					deriv_sigma_offset = this->deriv_sigma_buffer_->offset(j,i * NUM_GAUSS + k);
					deriv_mu1_offset = this->deriv_mu1_buffer_->offset(j,i * NUM_GAUSS + k);
					deriv_mu2_offset = this->deriv_mu2_buffer_->offset(j,i * NUM_GAUSS + k);

					//tmp_deriv_weight_offset = this->tmp_deriv_weight_buffer_.offset(0,k);
					tmp_deriv_weight_offset = deriv_weight_offset;
				}

				// precompute sigma^2
				Dtype sigma2 = sigma*sigma;

				Dtype gauss_sum = 0;

				int weights_index = 0;
				for (int y = 0; y < this->kernel_h_; ++y) {
					for (int x = 0; x < this->kernel_w_; ++x) {

						Dtype dist_x = x - mu1;
						Dtype dist_x_2 = dist_x*dist_x;

						Dtype dist_y = y - mu2;
						Dtype dist_y_2 = dist_y*dist_y;

						Dtype dist = dist_x_2 + dist_y_2;
						Dtype gauss_value = exp(-(dist)/(2 * sigma2));

						weight_tmp[weights_index] = w * gauss_value;
						//weight[weights_offset + weights_index] += w * gauss_value * 1.0 /(2 * 3.1416 * sigma2);
						if (is_backward_pass) {

							tmp_deriv_weight[tmp_deriv_weight_offset + weights_index] = gauss_value;
							deriv_mu1[deriv_mu1_offset + weights_index] = (dist_x / sigma2) * gauss_value;
							deriv_mu2[deriv_mu2_offset + weights_index] = (dist_y / sigma2) * gauss_value;
							deriv_sigma[deriv_sigma_offset + weights_index] = (dist / (sigma2 * sigma)) * gauss_value;
						}

						gauss_sum += gauss_value;

						++weights_index;
					}
				}

				// normalize current gauss and add it to final weight kernel buffer and add subfeature weight factor
				Dtype normalize_factor = (Dtype)1.0/gauss_sum;
				caffe_axpy(kernel_size, normalize_factor, weight_tmp, weight + weights_offset);

				if (is_backward_pass) {
					// normalize weight, mu1, mu2 and sigma kernels with the sum over gaussian
					// by definition we could use 1/(2*pi*sigma^2) but due to discretization error
					// we need to use actual sum
					//Dtype gauss_sum = 1.0 / caffe_cpu_asum(kernel_size, deriv_weight + deriv_weight_offset);
					Dtype gauss_mu1_sum_abs = caffe_cpu_asum(kernel_size, deriv_mu1 + deriv_mu1_offset);
					Dtype gauss_mu2_sum_abs = caffe_cpu_asum(kernel_size, deriv_mu2 + deriv_mu2_offset);
					Dtype gauss_sigma_sum_abs = caffe_cpu_asum(kernel_size, deriv_sigma + deriv_sigma_offset);

					Dtype gauss_mu1_sum = caffe_sum(kernel_size, deriv_mu1 + deriv_mu1_offset);
					Dtype gauss_mu2_sum = caffe_sum(kernel_size, deriv_mu2 + deriv_mu2_offset);
					Dtype gauss_sigma_sum = caffe_sum(kernel_size, deriv_sigma + deriv_sigma_offset);

					// add derivatives of normalization factor
					// i.e. G(x)/N(x) from GMM equation has derivate as N(x)*G'(x) - G(x) * N'(x)
					// where  G is guassian and N is sum of gaussian for normalization,
					// while G' is derivative of gaussian and N' is sum of derivative of gaussian
					gauss_mu1_sum = abs(gauss_mu1_sum) > 1e-10 ? gauss_mu1_sum : 0;
					gauss_mu2_sum = abs(gauss_mu2_sum) > 1e-10 ? gauss_mu2_sum : 0;

					caffe_cpu_axpby(kernel_size, -1* gauss_mu1_sum, tmp_deriv_weight + tmp_deriv_weight_offset, gauss_sum, deriv_mu1 + deriv_mu1_offset);
					caffe_cpu_axpby(kernel_size, -1* gauss_mu2_sum, tmp_deriv_weight + tmp_deriv_weight_offset, gauss_sum, deriv_mu2 + deriv_mu2_offset);
					caffe_cpu_axpby(kernel_size, -1* gauss_sigma_sum, tmp_deriv_weight + tmp_deriv_weight_offset, gauss_sum, deriv_sigma + deriv_sigma_offset);


					caffe_scal(kernel_size, (Dtype)1.0/gauss_sum, tmp_deriv_weight + tmp_deriv_weight_offset);
					caffe_scal(kernel_size, (Dtype)w/(gauss_sum*gauss_sum), deriv_mu1 + deriv_mu1_offset);
					caffe_scal(kernel_size, (Dtype)w/(gauss_sum*gauss_sum), deriv_mu2 + deriv_mu2_offset);
					caffe_scal(kernel_size, (Dtype)w/(gauss_sum*gauss_sum), deriv_sigma + deriv_sigma_offset);


				}

			}
			if (is_backward_pass) {


				// notice: deriv_error_buffer_ has first two dimensions switched to enable more efficent computation of bottom error in backward process
				int deriv_error_offset = this->deriv_error_buffer_->offset(j,i);

				int kernel_size = this->kernel_h_ * this->kernel_w_;
				for (int weights_index = 0; weights_index < kernel_size; ++weights_index) {
					// copy weights (GMM) into derivate/error weights which are the same but in reverse order
					deriv_error[deriv_error_offset + kernel_size - weights_index - 1] = weight[weights_offset + weights_index];
				}

			}
		}
	}

	free(weight_tmp);
}


template <typename Dtype>
void GaussianConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// forward CPU:
	// go over each bottom sub-feature i
	// go over each top feature j
	// go over K gaussians (K = 4)
	// convolve bottom feature with k-th gaussian 
	// and add values/matrix to j-th result
	// add bias b[j] to  j-th result

	// precompute kernels from gaussian parameters but only for forward pass
	this->precompute_guassian_weights(false);

	for (int i = 0; i < bottom.size(); ++i) {

		const Dtype* weight = this->weight_buffer_->cpu_data();
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* top_data = top[i]->mutable_cpu_data();

		for (int n = 0; n < this->num_; ++n) {
			this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
					top_data + top[i]->offset(n));
			if (this->bias_term_) {
				const Dtype* bias = this->param_buffer_bias_->cpu_data();
				this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
			}
		}
	}
}


template <typename Dtype>
void GaussianConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// backward CPU:
	{
		for (int i = 0; i < this->blobs_.size(); ++i) {
			Dtype* tt = this->blobs_[i]->mutable_cpu_data();
			caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_diff());
		}
	}
	int NUM_GAUSS_PER_AXIS = this->layer_param_.convolution_param().number_gauss();
	int NUM_GAUSS =  NUM_GAUSS_PER_AXIS * NUM_GAUSS_PER_AXIS;

	// precompute kernels from gaussian parameters for forward and backward pass
	this->precompute_guassian_weights(true);


	Blob<Dtype>& gauss_param_buffer_w = *this->param_buffer_w_;
	Blob<Dtype>& gauss_param_buffer_mu1 = *this->param_buffer_mu1_;
	Blob<Dtype>& gauss_param_buffer_mu2 = *this->param_buffer_mu2_;
	Blob<Dtype>& gauss_param_buffer_sigma = *this->param_buffer_sigma_;

	// clear values for gauassian parameters diffs
	if (this->param_propagate_down_[0]) {
		caffe_set(gauss_param_buffer_w.count(), Dtype(0), gauss_param_buffer_w.mutable_cpu_diff());
		caffe_set(gauss_param_buffer_mu1.count(), Dtype(0), gauss_param_buffer_mu1.mutable_cpu_diff());
		caffe_set(gauss_param_buffer_mu2.count(), Dtype(0), gauss_param_buffer_mu2.mutable_cpu_diff());
		caffe_set(gauss_param_buffer_sigma.count(), Dtype(0), gauss_param_buffer_sigma.mutable_cpu_diff());

	}

	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->cpu_diff();
		const Dtype* bottom_data = bottom[i]->cpu_data();

		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

		if (this->param_propagate_down_[0] || propagate_down[i]) {

			for (int n = 0; n < this->num_; ++n) {

				if (propagate_down[i]) {
					// calculate error matrices for lower layer
					// go over each top feature j
					// go over K gaussians (K = 4)
					// convolve top j-th error matrix with k-th gaussian
					// multiply result with bottom activation values (element wise)
					// save result to top_error[i,k]

					this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight_buffer_->cpu_data(),
				              	  	  	  	  bottom_diff + bottom[i]->offset(n));
				}
				if (this->param_propagate_down_[0]) {
					// calculate gradients for each gaussain parameter (weights, mean [x,y], sigma)
					// go over each top feature j
					// go over K gaussians (K = 4)
					// convolve top j-th error matrix with k-th gaussian
					// multiply result with bottom activation values (element wise)
					// save result to top_error[i,k]

					// convert bottom activation data into column array where convolution can be performed using matrix multiplication
					im2col_cpu(bottom_data + bottom[i]->offset(n),
							this->conv_in_channels_,
							this->conv_in_height_, this->conv_in_width_,
							this->kernel_h_, this->kernel_w_,
							this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_,
							this->col_buffer_.mutable_cpu_data());

					// compute convolution with bottom activation data (in column format) with kernels for weight derivative,
					// do dot-product of resut with top error and store final result to gaussian parameter diffs (i.e to diff of blobs_[0])
					compute_parameter_deriv(n, this->col_buffer_, *deriv_weight_buffer_, *top[i], gauss_param_buffer_w, 0);

					// do the same for means (mu1, mu1) and sigma
					compute_parameter_deriv(n, this->col_buffer_, *deriv_mu1_buffer_, *top[i], gauss_param_buffer_mu1, 0);
					compute_parameter_deriv(n, this->col_buffer_, *deriv_mu2_buffer_, *top[i], gauss_param_buffer_mu2, 0);
					compute_parameter_deriv(n, this->col_buffer_, *deriv_sigma_buffer_, *top[i], gauss_param_buffer_sigma, 0);

				}
			}
		}
	}
	if (this->use_gmm_weight_normalization) {

		const Dtype* param_w_buffer = gauss_param_buffer_w.cpu_data();
		Dtype* param_w_buffer_diff = gauss_param_buffer_w.mutable_cpu_diff();

		Dtype* param_w_sign = this->tmp_w_sign_.mutable_cpu_data();
		Dtype* param_w_fabs = this->tmp_w_fabs_.mutable_cpu_data();

		caffe_cpu_sign(gauss_param_buffer_w.count(), param_w_buffer, param_w_sign);
		caffe_cpu_fabs(gauss_param_buffer_w.count(), param_w_buffer, param_w_fabs);

		for (int f = 0; f < this->conv_out_channels_; ++f) {
			Dtype w_sum = 0;
			Dtype weighted_gradient_sum = 0;
			for (int s = 0; s < this->conv_in_channels_; ++s) {
				w_sum += caffe_cpu_asum(NUM_GAUSS, param_w_buffer + gauss_param_buffer_w.offset(0,s,f));

				// sum gradients of all subfeatures from the same feature weighted by individual GMM weights
				weighted_gradient_sum += caffe_cpu_dot(NUM_GAUSS,
															param_w_buffer + gauss_param_buffer_w.offset(0,s,f),
															param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f));
			}

			// normalize by sum of weights
			weighted_gradient_sum = weighted_gradient_sum / w_sum;


			for (int s = 0; s < this->conv_in_channels_; ++s) {

				// from abs
				caffe_mul(NUM_GAUSS, param_w_sign + this->tmp_w_sign_.offset(0,s,f),
									param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f),
									param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f));

				// subtract normalized weighted gradient sum from each gradient
				caffe_add_scalar(NUM_GAUSS, -1 * weighted_gradient_sum, param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f));

				// them multiply by sign(w)
				caffe_mul(NUM_GAUSS, param_w_sign + this->tmp_w_sign_.offset(0,s,f),
							param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f),
							param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f));

				// and finally add normalization factor
				caffe_scal(NUM_GAUSS, (Dtype)1./w_sum, param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f));

			}
		}
	}
	// add learning rate normalization based on size of input map and size of kernel
	// gradient(w) = gradient(w)/ (I * J )
	// gradient(mu1) = gradient(mu1) / (I * J )
	// gradient(mu2) = gradient(mu2) / (I * J )
	// gradient(sigma) = gradient(sigma) / (I * J)

	//Dtype norm_factor = 1.0 / (Dtype)(this->conv_in_height_ * this->conv_in_width_* this->kernel_h_ * this->kernel_w_);
	//Dtype norm_factor = 1.0 / (Dtype)(this->conv_in_height_ * this->conv_in_width_);
	Dtype norm_factor = 1.0 / (Dtype)this->conv_out_spatial_dim_; // THIS SHOULD BE CORRECT ONE !!!

	Dtype* gauss_w_deriv = gauss_param_buffer_w.mutable_cpu_diff();
	Dtype* gauss_mu1_deriv = gauss_param_buffer_mu1.mutable_cpu_diff();
	Dtype* gauss_mu2_deriv = gauss_param_buffer_mu2.mutable_cpu_diff();
	Dtype* gauss_sigma_deriv = gauss_param_buffer_sigma.mutable_cpu_diff();

	// add default normalizing factor to all parameters
	caffe_scal(gauss_param_buffer_w.count(), norm_factor, gauss_w_deriv);
	caffe_scal(gauss_param_buffer_mu1.count(), norm_factor, gauss_mu1_deriv);
	caffe_scal(gauss_param_buffer_mu2.count(), norm_factor, gauss_mu2_deriv);
	caffe_scal(gauss_param_buffer_sigma.count(), norm_factor, gauss_sigma_deriv);
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::compute_parameter_deriv(int num_iter, 
		const Blob<Dtype>& col_activations_buffer, const Blob<Dtype>& deriv_kernels_buffer,
		const Blob<Dtype>& top_error_buffer,
		Blob<Dtype>& param_output_buffer, int param_output_offset) {

	int NUM_GAUSS_PER_AXIS = this->layer_param_.convolution_param().number_gauss();
	int NUM_GAUSS =  NUM_GAUSS_PER_AXIS * NUM_GAUSS_PER_AXIS;

	const Dtype* col_buff = col_activations_buffer.cpu_data();
	const Dtype* deriv_kernels = deriv_kernels_buffer.cpu_data();
	const Dtype* top_error = top_error_buffer.cpu_diff(); // using diff !!

	Dtype* tmp_buff = tmp_buffer_.mutable_cpu_data();

	Dtype* param_output = param_output_buffer.mutable_cpu_diff(); // using diff !!

	for (int s = 0; s < this->conv_in_channels_; ++s) {

		// compute convolution of activations with weight deriv gaussian for all s-th sub-feature (i.e. NUM_GAUSS * NUM_OUT_CHANNELS number of convolutions)
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
				this->conv_out_channels_ * NUM_GAUSS,
				this->conv_out_spatial_dim_,
				this->kernel_h_ * this->kernel_w_,
				(Dtype)1., deriv_kernels + deriv_kernels_buffer.offset(s) , col_buff + col_activations_buffer.offset(s),
				(Dtype)0., tmp_buff);

		// perform dot-product of each convolution result with top error diff for each feature individualy
		for (int f = 0; f < this->conv_out_channels_; ++f) {

			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
					NUM_GAUSS,
					1,
					this->conv_out_spatial_dim_,
					(Dtype)1., tmp_buff + tmp_buffer_.offset(f), top_error + top_error_buffer.offset(num_iter,f),
					(Dtype)1., param_output + param_output_buffer.offset(param_output_offset, s, f));
		}
	}
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Forward_cpu(bottom, top);
}
template <typename Dtype>
void GaussianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Backward_gpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(GaussianConvLayer);
REGISTER_LAYER_CLASS(GaussianConv);

}  // namespace caffe
