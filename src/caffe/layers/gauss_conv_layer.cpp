#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_extra.hpp"
#include "caffe/layers/gauss_conv_layer.hpp"

#include <ctime>
#include <algorithm>

#include <opencv/cv.hpp>

#define NUM_GAUSS_COMPONENT_PARAM 4
#define NUM_GAUSS_PARAM 1

namespace caffe {

template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	//openblas_set_num_threads(1);
	CHECK_GT(this->layer_param_.convolution_param().number_gauss_size(),0) << "Missing at least one number_gauss parameter.";

	int NUM_GAUSS_PER_AXIS_X = this->layer_param_.convolution_param().number_gauss(0);
	int NUM_GAUSS_PER_AXIS_Y = this->layer_param_.convolution_param().number_gauss_size() > 1? this->layer_param_.convolution_param().number_gauss(1) : NUM_GAUSS_PER_AXIS_X;
	NUM_GAUSS =  NUM_GAUSS_PER_AXIS_X * NUM_GAUSS_PER_AXIS_Y;

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
	// Setup dilation dimensions (dilation_).
	this->dilation_.Reshape(spatial_dim_blob_shape);
	int* dilation_data = this->dilation_.mutable_cpu_data();
	const int num_dilation_dims = conv_param.dilation_size();
	CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
		num_dilation_dims == this->num_spatial_axes_)
	  << "dilation must be specified once, or once per spatial dimension "
	  << "(dilation specified " << num_dilation_dims << " times; "
	  << this->num_spatial_axes_ << " spatial dims).";
	const int kDefaultDilation = 1;
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
	dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
					   conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
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

	this->gmm_component_border_bound = this->layer_param_.convolution_param().gmm_component_border_bound();
	this->gmm_sigma_lower_bound = this->layer_param_.convolution_param().gmm_sigma_lower_bound();

	// Handle the parameters: weights and biases.
	// - blobs_[0] holds the filter weights
	// - blobs_[1] holds the biases (optional)
	this->bias_term_ = this->layer_param_.convolution_param().bias_term();
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {

		this->num_extra_blobs = 8;

		// to support loading older snapshots we need to ignore last two blobs which hold random means for components merging
		if (this->layer_param_.convolution_param().gmm_legacy_merge_blobs()) {
			LOG(INFO) << "Using legacy loading support that does not have merge blobs";
			this->num_extra_blobs -= 2;
		} else {
			LOG(INFO) << "Legacy loading support for merge blobs disabled";
		}
		if (this->bias_term_) {
			this->blobs_.resize(1 + NUM_GAUSS_COMPONENT_PARAM + NUM_GAUSS_PARAM + this->num_extra_blobs );
		} else {
			this->blobs_.resize(NUM_GAUSS_COMPONENT_PARAM +  NUM_GAUSS_PARAM + this->num_extra_blobs );
		}
		// Initialize and fill the weights:
		// output channels x input channels per-group x kernel height x kernel width
		// set size of all per-components GMM paramteres (weight,mu1,mu2,sigma)
		int blobs_index = 0;
		for (int i = 0; i < NUM_GAUSS_COMPONENT_PARAM; i++)
			this->blobs_[blobs_index++].reset(new Blob<Dtype>(1, this->conv_in_channels_, NUM_GAUSS, this->conv_out_channels_));

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

		Blob<Dtype> tmp_w(1, this->conv_in_channels_,  NUM_GAUSS, this->conv_out_channels_);
		Blob<Dtype> tmp_sigma(1, this->conv_in_channels_, NUM_GAUSS, this->conv_out_channels_);
		Blob<Dtype> tmp_mu1(1, this->conv_in_channels_, NUM_GAUSS, this->conv_out_channels_);
		Blob<Dtype> tmp_mu2(1, this->conv_in_channels_, NUM_GAUSS, this->conv_out_channels_);

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
		Dtype* offset_x = new Dtype[NUM_GAUSS_PER_AXIS_X];
		Dtype* offset_y = new Dtype[NUM_GAUSS_PER_AXIS_Y];

		// use gmm_component_border_bound as start and stop position of where components are allowed to be within the kernel
		Dtype gmm_kernel_h_ = (Dtype)this->kernel_h_ - 2*this->gmm_component_border_bound;
		Dtype gmm_kernel_w_ = (Dtype)this->kernel_w_ - 2*this->gmm_component_border_bound;

		for (int i = 0; i < NUM_GAUSS_PER_AXIS_X; i++) {
			offset_x[i] = this->gmm_component_border_bound + (i)*gmm_kernel_w_ /(Dtype)(NUM_GAUSS_PER_AXIS_X) + (- 0.5+(gmm_kernel_w_)/(Dtype)(2*NUM_GAUSS_PER_AXIS_X));
		}
		for (int i = 0; i < NUM_GAUSS_PER_AXIS_Y; i++) {
			offset_y[i] = this->gmm_component_border_bound + (i)*gmm_kernel_h_ /(Dtype)(NUM_GAUSS_PER_AXIS_Y) + (- 0.5+(gmm_kernel_h_)/(Dtype)(2*NUM_GAUSS_PER_AXIS_Y));
		}

		// add offset to mean so that (0,0) is at center
		//int kernel_center_w = this->kernel_w_ / 2;
		//int kernel_center_h = this->kernel_h_ / 2;
		int kernel_center_w = 0;
		int kernel_center_h = 0;

		const int outer_size = this->conv_in_channels_;
		const int middle_size = NUM_GAUSS;
		const int inner_size = this->conv_out_channels_;

		for (int i1 = 0; i1 < outer_size; ++i1) {
			for (int i2 = 0; i2 < middle_size; ++i2) {
				for (int i3 = 0; i3 < inner_size; ++i3) {
					const int gauss_idx = i2;
					const int offset_idx = (i1 * middle_size + i2 )* inner_size + i3;
					mu1_buf[offset_idx] = offset_x[gauss_idx / NUM_GAUSS_PER_AXIS_Y] - kernel_center_w;
					mu2_buf[offset_idx] = offset_y[gauss_idx %  NUM_GAUSS_PER_AXIS_Y] - kernel_center_h;
				}
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


		this->param_buffer_mu1_discr_.reset(new Blob<Dtype>());
		this->param_buffer_mu2_discr_.reset(new Blob<Dtype>());

		int precomputed_blobs_offset = NUM_GAUSS_COMPONENT_PARAM + NUM_GAUSS_PARAM;
		if (this->bias_term_)
			precomputed_blobs_offset++;

		this->weight_buffer_.reset(new Blob<Dtype>());
		this->weight_vert_buffer_.reset(new Blob<Dtype>());
		this->weight_horiz_buffer_.reset(new Blob<Dtype>());
		this->deriv_error_buffer_.reset(new Blob<Dtype>());
		this->deriv_weight_buffer_.reset(new Blob<Dtype>());
		this->deriv_sigma_buffer_.reset(new Blob<Dtype>());
		this->deriv_mu1_buffer_.reset(new Blob<Dtype>());
		this->deriv_mu2_buffer_.reset(new Blob<Dtype>());

		this->random_mu1_buffer_.reset(new Blob<Dtype>());
		this->random_mu2_buffer_.reset(new Blob<Dtype>());

		this->blobs_[precomputed_blobs_offset + 0] = this->weight_buffer_;
		this->blobs_[precomputed_blobs_offset + 1] = this->deriv_error_buffer_;
		this->blobs_[precomputed_blobs_offset + 2] = this->deriv_weight_buffer_;
		this->blobs_[precomputed_blobs_offset + 3] = this->deriv_sigma_buffer_;
		this->blobs_[precomputed_blobs_offset + 4] = this->deriv_mu1_buffer_;
		this->blobs_[precomputed_blobs_offset + 5] = this->deriv_mu2_buffer_;

		if (this->layer_param_.convolution_param().gmm_legacy_merge_blobs() == false) {
			this->blobs_[precomputed_blobs_offset + 6] = this->random_mu1_buffer_;
			this->blobs_[precomputed_blobs_offset + 7] = this->random_mu2_buffer_;
			// NOTE: adding/removing blobs here also needs to be taken care of in CuDNNGaussianConvLayer

			// NOTICE: make sure lr_param for random mean buffers are set to 1 so that they are synced between multiple GPUs
			if (this->layer_param().param_size() > precomputed_blobs_offset + 6){
				CHECK_EQ(1, this->layer_param().param(precomputed_blobs_offset + 6).lr_mult()) << "lr_param for random mean buffer (index " << precomputed_blobs_offset + 6 << ") must be set to 1";
				CHECK_EQ(0, this->layer_param().param(precomputed_blobs_offset + 6).decay_mult())  << "decay_mult for random mean buffer (index " << precomputed_blobs_offset + 6 << ") must be set to 0";

				// by default decay_mult is set to 1, however multiply all random values will be canceled out because component merging re-scale values to desired size
			}
			if (this->layer_param().param_size() > precomputed_blobs_offset + 7){
				CHECK_EQ(1, this->layer_param().param(precomputed_blobs_offset + 7).lr_mult())  << "lr_param for random mean buffer (index " << precomputed_blobs_offset + 7 << ") must be set to 1";
				CHECK_EQ(0, this->layer_param().param(precomputed_blobs_offset + 7).decay_mult())  << "decay_mult for random mean buffer (index " << precomputed_blobs_offset + 7 << ") must be set to 0";

				// by default decay_mult is set to 1, however multiply all random values will be canceled out because component merging re-scale values to desired size
			}
		}
	}
	// Propagate gradients to the parameters (as directed by backward pass).
	this->param_propagate_down_.resize(this->blobs_.size(), true);

	// decide if needed to perform gmm weight normalization
	this->use_gmm_weight_normalization = this->layer_param_.convolution_param().gmm_weight_normalization();

	// decide if needed to perform gmm gauss normalization
	this->use_gmm_gauss_normalization = this->layer_param_.convolution_param().gmm_gauss_normalization();
	this->use_gmm_square_gauss_normalization = this->layer_param_.convolution_param().gmm_square_gauss_normalization();
	this->gmm_mean_iteration_step = this->layer_param_.convolution_param().gmm_mean_iteration_step();
	this->gmm_sigma_iteration_step = this->layer_param_.convolution_param().gmm_sigma_iteration_step();

	// make sure component merging is done only at training step
	if (this->phase_ == TRAIN)
		this->gmm_merge_iteration_step = this->layer_param_.convolution_param().gmm_merge_iteration_step();
	else
		this->gmm_merge_iteration_step = 0;

	this->gmm_merge_threshold = this->layer_param_.convolution_param().gmm_merge_threshold();

	this->use_gmm_seperable_kernels = this->layer_param_.convolution_param().gmm_seperable_forward_pass();

	this->gmm_discretize_mean = this->layer_param_.convolution_param().gmm_discretize_mean();


	this->current_iteration_index = 0;

	// setup default bottom/top dimensions to zero
	this->bottom_dim_ = 0;
	this->top_dim_ = 0;
	this->num_ = 0;
}

template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


	const int first_spatial_axis = this->channel_axis_ + 1;
	//int NUM_GAUSS_PER_AXIS = this->layer_param_.convolution_param().number_gauss();
	//int NUM_GAUSS =  NUM_GAUSS_PER_AXIS * NUM_GAUSS_PER_AXIS;

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
	this->compute_output_shape();
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

	this->param_buffer_mu1_discr_->ReshapeLike(*this->param_buffer_mu1_);
	this->param_buffer_mu2_discr_->ReshapeLike(*this->param_buffer_mu2_);


	this->random_mu1_buffer_->ReshapeLike(*this->param_buffer_w_);
	this->random_mu2_buffer_->ReshapeLike(*this->param_buffer_w_);

	this->weight_buffer_->Reshape(this->conv_out_channels_, this->conv_in_channels_, this->kernel_h_, this->kernel_w_);

	// seperable kernels
	this->weight_vert_buffer_->Reshape(this->conv_out_channels_, NUM_GAUSS, this->conv_in_channels_, this->kernel_h_);
	this->weight_horiz_buffer_->Reshape(this->conv_out_channels_, NUM_GAUSS, this->conv_in_channels_, this->kernel_w_);

	this->deriv_error_buffer_->Reshape(this->conv_in_channels_, this->conv_out_channels_, this->kernel_h_, this->kernel_w_);
	this->deriv_weight_buffer_->Reshape(this->conv_in_channels_, NUM_GAUSS, this->conv_out_channels_, this->kernel_h_ * this->kernel_w_);
	this->deriv_sigma_buffer_->Reshape(this->conv_in_channels_, NUM_GAUSS, this->conv_out_channels_, this->kernel_h_ * this->kernel_w_);
	this->deriv_mu1_buffer_->Reshape(this->conv_in_channels_, NUM_GAUSS, this->conv_out_channels_, this->kernel_h_ * this->kernel_w_);
	this->deriv_mu2_buffer_->Reshape(this->conv_in_channels_, NUM_GAUSS, this->conv_out_channels_, this->kernel_h_ * this->kernel_w_);

	this->is_weight_enabled_buffer_.Reshape(this->conv_out_channels_, NUM_GAUSS, this->conv_in_channels_, 1);

	this->guass_dist_buffer_.ReshapeLike(*this->deriv_weight_buffer_);
	this->gauss_dist_square_buffer_.ReshapeLike(*this->deriv_weight_buffer_);
	this->deriv_mu1_times_gauss_dist_buffer_.ReshapeLike(*this->deriv_weight_buffer_);
	this->deriv_mu2_times_gauss_dist_buffer_.ReshapeLike(*this->deriv_weight_buffer_);
	this->deriv_sigma_times_gauss_dist_buffer_.ReshapeLike(*this->deriv_weight_buffer_);

	this->guass_norm_buffer_.ReshapeLike(*this->param_buffer_w_);
	this->deriv_mu1_sums_buffer_.ReshapeLike(*this->param_buffer_w_);
	this->deriv_mu2_sums_buffer_.ReshapeLike(*this->param_buffer_w_);
	this->deriv_sigma_sums_buffer_.ReshapeLike(*this->param_buffer_w_);

	// pre-computed offset indexes for batched sums (when using caffe_gpu_sum)
	this->create_precompute_index(this->tmp_precomp_index_, this->conv_in_channels_ * NUM_GAUSS * this->conv_out_channels_, this->kernel_h_ * this->kernel_w_);

	this->tmp_deriv_weight_buffer_.Reshape(1, NUM_GAUSS , this->kernel_h_, this->kernel_w_);

	this->param_buffer_sigma_square_inv_.ReshapeLike(*this->param_buffer_sigma_);
	this->param_buffer_sigma_cube_inv_.ReshapeLike(*this->param_buffer_sigma_);
	this->param_buffer_sigma_square_inv_half_.ReshapeLike(*this->param_buffer_sigma_);

	// Set up the all ones "bias multiplier" for adding biases by BLAS
	if (this->bias_term_) {
		vector<int> bias_multiplier_shape(1, this->height_out_ * this->width_out_);
		this->bias_multiplier_.Reshape(bias_multiplier_shape);
		caffe_set(this->bias_multiplier_.count(), Dtype(1), this->bias_multiplier_.mutable_cpu_data());
	}
}

template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::create_precompute_index(Blob<int>& precomp_index_buff, const int index_size, const int kernel_size) {

	precomp_index_buff.Reshape(1, 1, 1, index_size + 1);

	int* tmp_precomp_index_cpu = precomp_index_buff.mutable_cpu_data();

	tmp_precomp_index_cpu[0] = 0;

	for (int i = 0; i < precomp_index_buff.count()-1; i++)
		tmp_precomp_index_cpu[i+1] = kernel_size * (i+1);

}


template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::compute_output_shape() {
	this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
    		  / this->stride_h_ + 1;
	this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
    		  / this->stride_w_ + 1;
}

template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)  {
	LOG(ERROR) << "BaseGaussianConvLayer does not have Forward_cpu implementation";
	throw std::exception();
}
template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LOG(ERROR) << "BaseGaussianConvLayer does not have Forward_gpu implementation";
	throw std::exception();
}
template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	LOG(ERROR) << "BaseGaussianConvLayer does not have Backward_cpu implementation";
	throw std::exception();
}
template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	LOG(ERROR) << "BaseGaussianConvLayer does not have Backward_gpu implementation";
	throw std::exception();
}

////////////////////////////////////////////////

template <typename Dtype>
GaussianConvLayer<Dtype>::~GaussianConvLayer() {

	  if (A != NULL) delete A;
	  if (B != NULL) delete B;
	  if (C != NULL) delete C;

#ifndef CPU_ONLY
	  if (d_A != NULL) cudaFree(d_A);
	  if (d_B != NULL) cudaFree(d_B);
	  if (d_C != NULL) cudaFree(d_C);

#endif

}

template <typename Dtype>
void GaussianConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	BaseGaussianConvLayer<Dtype>::LayerSetUp(bottom, top);

}


template <typename Dtype>
void GaussianConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	BaseGaussianConvLayer<Dtype>::Reshape(bottom, top);

	const int first_spatial_axis = this->channel_axis_ + 1;

	// The im2col result buffer will only hold one image at a time to avoid
	// overly large memory usage. In the special case of 1x1 convolution
	// it goes lazily unused to save memory.
	if (this->reverse_dimensions()) {
		this->col_buffer_.Reshape(this->conv_in_channels_, this->kernel_h_ * this->kernel_w_, this->height_, this->width_);
	} else {
		this->col_buffer_.Reshape(this->conv_in_channels_, this->kernel_h_ * this->kernel_w_, this->height_out_, this->width_out_);
	}

	this->tmp_blob_.Reshape(1, this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_);

	this->tmp_buffer_sepe_1_.Reshape(this->conv_in_channels_, this->NUM_GAUSS* this->conv_out_channels_, this->height_ + 2* this->pad_h_, this->width_ + 2* this->pad_w_);
	this->tmp_buffer_sepe_2_.Reshape(this->conv_in_channels_, this->NUM_GAUSS* this->conv_out_channels_, this->height_ + 2* this->pad_h_, this->width_ + 2* this->pad_w_);

	//this->tmp_buffer_.Reshape(this->conv_in_channels_, this->conv_out_channels_, NUM_GAUSS, this->height_out_ * this->width_out_);
	this->tmp_buffer_.Reshape(this->conv_in_channels_, this->NUM_GAUSS, this->conv_out_channels_, this->height_out_ * this->width_out_);

	// pre-computed offset indexes for batched sums (when using caffe_gpu_sum)
	this->tmp_index_.Reshape(1, 1, 1, this->conv_in_channels_ * this->NUM_GAUSS * this->conv_out_channels_ + 1);

	int* tmp_index_cpu = this->tmp_index_.mutable_cpu_data();

	tmp_index_cpu[0] = 0;

	for (int i = 0; i < this->tmp_index_.count()-1; i++) tmp_index_cpu[i+1] = this->height_out_ * this->width_out_*(i+1);

	tmp_index_gpu = this->tmp_index_.mutable_gpu_data();



	this->accum_bottom_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());

	if (A != NULL) delete A;
	if (B != NULL) delete B;
	if (C != NULL) delete C;

    A = new const Dtype*[this->conv_out_channels_ * this->conv_in_channels_];
	B = new const Dtype*[this->conv_out_channels_ * this->conv_in_channels_];
	C = new Dtype*[this->conv_out_channels_ * this->conv_in_channels_];

#ifndef CPU_ONLY
	if (d_A != NULL) cudaFree(d_A);
	if (d_B != NULL) cudaFree(d_B);
	if (d_C != NULL) cudaFree(d_C);

	cudaMalloc((void**)&d_A, this->conv_out_channels_ * this->conv_in_channels_ * sizeof(Dtype*));
	cudaMalloc((void**)&d_B, this->conv_out_channels_ * this->conv_in_channels_ * sizeof(Dtype*));
	cudaMalloc((void**)&d_C, this->conv_out_channels_ * this->conv_in_channels_ * sizeof(Dtype*));
#endif


	this->tmp_w_sign_.ReshapeLike(*this->param_buffer_w_);
	this->tmp_w_fabs_.ReshapeLike(*this->param_buffer_w_);

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
void BaseGaussianConvLayer<Dtype>::precompute_guassian_weights(bool is_backward_pass) {
    do_precompute_guassian_weights(*this->param_buffer_w_.get(),
                                   *this->param_buffer_mu1_.get(),
                                   *this->param_buffer_mu2_.get(),
                                   *this->param_buffer_sigma_.get(),
                                   this->conv_in_channels_, this->conv_out_channels_, this->NUM_GAUSS,
                                   this->kernel_h_, this->kernel_w_,
                                   is_backward_pass,
                                   this->use_gmm_weight_normalization,
                                   this->use_gmm_square_gauss_normalization,
                                   this->gmm_sigma_lower_bound,
                                   this->gmm_component_border_bound,
                                   this->gmm_discretize_mean,
                                   this->weight_buffer_.get(),
                                   this->weight_vert_buffer_.get(),
                                   this->weight_horiz_buffer_.get(),
								   &this->is_weight_enabled_buffer_,
                                   this->deriv_error_buffer_.get(),
                                   this->deriv_weight_buffer_.get(),
                                   this->deriv_mu1_buffer_.get(),
                                   this->deriv_mu2_buffer_.get(),
                                   this->deriv_sigma_buffer_.get());
}

template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::do_precompute_guassian_weights(Blob<Dtype>& gauss_param_buffer_w,
                                                                  Blob<Dtype>& gauss_param_buffer_mu1,
                                                                  Blob<Dtype>& gauss_param_buffer_mu2,
                                                                  Blob<Dtype>& gauss_param_buffer_sigma,
                                                                  int num_in_channels, int num_out_channels, int num_gauss,
                                                                  int kernel_h, int kernel_w,
                                                                  bool is_backward_pass,
                                                                  bool use_gmm_weight_normalization,
                                                                  bool use_gmm_square_gauss_normalization,
                                                                  bool gmm_discretize_mean,
                                                                  Dtype gmm_sigma_lower_bound,
                                                                  Dtype gmm_component_border_bound,
                                                                  // output buffers
                                                                  Blob<Dtype>* weight_buffer,
                                                                  Blob<Dtype>* weight_vert_buffer,
                                                                  Blob<Dtype>* weight_horiz_buffer,

																  Blob<int>* is_weight_enabled_buffer,
                                                                  Blob<Dtype>* deriv_error_buffer,
                                                                  Blob<Dtype>* deriv_weight_buffer,
                                                                  Blob<Dtype>* deriv_mu1_buffer,
                                                                  Blob<Dtype>* deriv_mu2_buffer,
                                                                  Blob<Dtype>* deriv_sigma_buffer) {

    /*

    // input buffers

    param_buffer_w_;
	param_buffer_mu1_;
	param_buffer_mu2_;
	param_buffer_sigma_;

    is_weight_enabled_buffer_

    // input size values:

    conv_out_channels_
    conv_in_channels_
    NUM_GAUSS

    kernel_h_
    kernel_w_


    // switches/flags
    is_backward_pass
    use_gmm_weight_normalization
    use_gmm_square_gauss_normalization

    gmm_sigma_lower_bound
    gmm_component_border_bound
    gmm_discretize_mean

    // output buffers
    weight_buffer_
    weight_vert_buffer_
    weight_horiz_buffer_

    deriv_error_buffer_
    deriv_weight_buffer_
    deriv_mu1_buffer_
    deriv_mu2_buffer_
    deriv_sigma_buffer_

    */

    // read number of channels, number of output_features and number of gaussians from parameters


	clock_t start_t = clock();

	// force head to CPU to avoid copying from GPU back to CPU which would be zeroed anyway
	weight_buffer->force_cpu_data();

	if (weight_vert_buffer != NULL) weight_vert_buffer->force_cpu_data();
	if (weight_horiz_buffer != NULL) weight_horiz_buffer->force_cpu_data();

	Dtype* weight = weight_buffer->mutable_cpu_data();

	Dtype* weight_vert = weight_vert_buffer != NULL ? weight_vert_buffer->mutable_cpu_data() : NULL;
	Dtype* weight_horiz = weight_horiz_buffer != NULL ? weight_horiz_buffer->mutable_cpu_data() : NULL;

	int* is_weight_enabled = is_weight_enabled_buffer != NULL ? is_weight_enabled_buffer->mutable_cpu_data() : NULL;

	Dtype* deriv_error, *deriv_weight, *deriv_sigma, *deriv_mu1, *deriv_mu2, *tmp_deriv_weight;


	if (is_backward_pass) {

		// force head to CPU to avoid copying from GPU back to CPU which would be zeroed anyway
		deriv_error_buffer->force_cpu_data();
		deriv_weight_buffer->force_cpu_data();
		deriv_mu1_buffer->force_cpu_data();
		deriv_mu2_buffer->force_cpu_data();
		deriv_sigma_buffer->force_cpu_data();

		deriv_error = deriv_error_buffer->mutable_cpu_data();
		deriv_weight = deriv_weight_buffer->mutable_cpu_data();
		deriv_mu1 = deriv_mu1_buffer->mutable_cpu_data();
		deriv_mu2 = deriv_mu2_buffer->mutable_cpu_data();
		deriv_sigma = deriv_sigma_buffer->mutable_cpu_data();

		//tmp_deriv_weight = this->tmp_deriv_weight_buffer_.mutable_cpu_data();
		tmp_deriv_weight = deriv_weight;

		memset(deriv_error, 0, sizeof(Dtype) * deriv_error_buffer->count());
		memset(deriv_weight, 0, sizeof(Dtype) * deriv_weight_buffer->count());
		memset(deriv_mu1, 0, sizeof(Dtype) * deriv_mu1_buffer->count());
		memset(deriv_mu2, 0, sizeof(Dtype) * deriv_mu2_buffer->count());
		memset(deriv_sigma, 0, sizeof(Dtype) * deriv_sigma_buffer->count());

		memset(tmp_deriv_weight, 0, sizeof(Dtype) * this->tmp_deriv_weight_buffer_.count());
	}

	memset(weight, 0, sizeof(Dtype) * weight_buffer->count());

	// pre-compute weights from guassian params
	/*Blob<Dtype>& gauss_param_buffer_w = *param_buffer_w;
	Blob<Dtype>& gauss_param_buffer_mu1 = *param_buffer_mu1;
	Blob<Dtype>& gauss_param_buffer_mu2 = *param_buffer_mu2;
	Blob<Dtype>& gauss_param_buffer_sigma = *param_buffer_sigma;*/

	const Dtype* gauss_params_w = gauss_param_buffer_w.cpu_data();
	Dtype* gauss_params_mu1 = gauss_param_buffer_mu1.mutable_cpu_data();
	Dtype* gauss_params_mu2 = gauss_param_buffer_mu2.mutable_cpu_data();
	Dtype* gauss_params_sigma = gauss_param_buffer_sigma.mutable_cpu_data();

	//int kernel_center_w = kernel_w / 2;
	//int kernel_center_h = kernel_h / 2;
	int kernel_center_w = 0;
	int kernel_center_h = 0;

    int kernel_size = kernel_h * kernel_w;

	Dtype* weight_tmp = (Dtype*)malloc(sizeof(Dtype)*kernel_size);

	for (int i = 0; i < num_out_channels; ++i) { // over each feature (i.e out feature)

		// new weight normalization over all subfeatures
		Dtype w_sum = 0;
		if (use_gmm_weight_normalization) {
			CHECK_EQ(0,1) << "GMM weight normalization not implemented with new version!!";

			for (int j = 0; j < num_in_channels; ++j) // sum weigths over all subfeature of feature f
				w_sum += caffe_cpu_asum(num_gauss, gauss_params_w + gauss_param_buffer_w.offset(0,j,i)); // TODO: fix S and F indexes
		} else {
			// set sum to 1 if we do not do weight normalization
			w_sum = 1;
		}

		Dtype w_sum_2 = w_sum*w_sum;

		for (int j = 0; j < num_in_channels; ++j) { // over each subfeature (i.e. in feature)

			int weights_offset = weight_buffer->offset(i,j);

			// original weight normalization within subfeature only
			//Dtype w_sum = caffe_cpu_asum(num_gauss, gauss_params_w + gauss_param_buffer_w.offset(0,j,i));
			//Dtype w_sum_2 = w_sum*w_sum;

			for (int k = 0; k < num_gauss; ++k) {
				Dtype w = 	gauss_params_w[gauss_param_buffer_w.offset(0,j,k,i)];
				//Dtype& mu1 = 	gauss_params_mu1[gauss_param_buffer_mu1.offset(0,j,k,i)] + kernel_center_w;
				//Dtype& mu2 = 	gauss_params_mu2[gauss_param_buffer_mu2.offset(0,j,k,i)] + kernel_center_h;
				Dtype& mu1_org = 	gauss_params_mu1[gauss_param_buffer_mu1.offset(0,j,k,i)];
				Dtype& mu2_org = 	gauss_params_mu2[gauss_param_buffer_mu2.offset(0,j,k,i)];
				Dtype& sigma = 	gauss_params_sigma[gauss_param_buffer_sigma.offset(0,j,k,i)];

				// do not allow sigma bellow 0.1 threshold !!
				sigma = std::max(gmm_sigma_lower_bound,sigma);

				// do not allow mean outside of kernel bounds reduced by gmm_component_border_bound
				mu1_org = std::max((Dtype)gmm_component_border_bound,mu1_org);
				mu2_org = std::max((Dtype)gmm_component_border_bound,mu2_org);

				mu1_org = std::min(kernel_w-1 - (Dtype)gmm_component_border_bound,mu1_org);
				mu2_org = std::min(kernel_h-1 - (Dtype)gmm_component_border_bound,mu2_org);

				Dtype mu1 = mu1_org;
				Dtype mu2 = mu2_org;

				// discretize mean
				if (gmm_discretize_mean) {
					// round means to integer values, however this should not be saved back to blob
					mu1 = round(mu1);
					mu2 = round(mu2);
				}

				// no need to compute kernel from this gaussian component if its weight is zero
				int is_valid_kernel = std::abs<Dtype>(w) > 0 ? 1 : 0;

				if (is_weight_enabled != NULL)
					is_weight_enabled[is_weight_enabled_buffer->offset(i,k,j)] = is_valid_kernel;

				if (is_valid_kernel == 0)
					continue;


				Dtype w_org = w;
				w = w/w_sum;

				int deriv_error_offset,deriv_weight_offset,deriv_sigma_offset,deriv_mu1_offset,deriv_mu2_offset, tmp_deriv_weight_offset;

				if (is_backward_pass) {
					// notice: deriv_error_buffer_ has first two dimensions switched to enable more efficent computation of derivates in backward process
					deriv_weight_offset = deriv_weight_buffer->offset(j, k, i);
					deriv_sigma_offset = deriv_sigma_buffer->offset(j, k, i);
					deriv_mu1_offset = deriv_mu1_buffer->offset(j, k, i);
					deriv_mu2_offset = deriv_mu2_buffer->offset(j, k, i);

					//tmp_deriv_weight_offset = this->tmp_deriv_weight_buffer_.offset(0,k);
					tmp_deriv_weight_offset = deriv_weight_offset;
				}

				// precompute sigma^2
				Dtype sigma2 = sigma*sigma;
				Dtype sigma3 = sigma2*sigma;

				Dtype sigma2_inv = 1/sigma2;
				Dtype sigma3_inv = 1/sigma3;

				Dtype sigma2_inv_half = sigma2_inv/2;

				Dtype gauss_sum = 0;
				Dtype gauss_square_sum = 0;

				int weights_index = 0;
				for (int y = 0; y < kernel_h; ++y) {
					for (int x = 0; x < kernel_w; ++x) {

						Dtype dist_x = x - mu1;
						Dtype dist_x_2 = dist_x*dist_x;

						Dtype dist_y = y - mu2;
						Dtype dist_y_2 = dist_y*dist_y;

						Dtype dist = dist_x_2 + dist_y_2;
						Dtype gauss_value = exp( -dist * sigma2_inv_half);

						weight_tmp[weights_index] = w * gauss_value;
						if (is_backward_pass) {

							tmp_deriv_weight[tmp_deriv_weight_offset + weights_index] = gauss_value;
							deriv_mu1[deriv_mu1_offset + weights_index] = (dist_x * sigma2_inv) * gauss_value;
							deriv_mu2[deriv_mu2_offset + weights_index] = (dist_y * sigma2_inv) * gauss_value;
							deriv_sigma[deriv_sigma_offset + weights_index] = (dist * sigma3_inv) * gauss_value;
						}

						gauss_sum += gauss_value;
						gauss_square_sum += gauss_value*gauss_value;

						++weights_index;
					}
				}


				if (use_gmm_gauss_normalization == false) {
					gauss_sum = 1;
					gauss_square_sum = 1;
				}
				Dtype norm_factor = gauss_sum;
				if (use_gmm_square_gauss_normalization) {
					norm_factor = gauss_square_sum;
				}

				// normalize current gauss and add it to final weight kernel buffer and add subfeature weight factor
				Dtype normalize_factor_inv = (Dtype)1.0/norm_factor;

				//Dtype normalize_factor = (Dtype)1.0/gauss_sum; // wrong version for normalization with gauss square - but it worked !!
				caffe_axpy(kernel_size, normalize_factor_inv, weight_tmp, weight + weights_offset);

				if (is_backward_pass) {
					// normalize weight, mu1, mu2 and sigma kernels with the sum over gaussian
					// by definition we could use 1/(2*pi*sigma^2) but due to discretization error
					// we need to use actual sum
					Dtype gauss_mu1_sum, gauss_mu2_sum, gauss_sigma_sum;
					if (use_gmm_square_gauss_normalization == false) {
						//gauss_sum = 1.0 / caffe_cpu_asum(kernel_size, deriv_weight + deriv_weight_offset);
						//gauss_mu1_sum_abs = caffe_cpu_asum(kernel_size, deriv_mu1 + deriv_mu1_offset);
						//gauss_mu2_sum_abs = caffe_cpu_asum(kernel_size, deriv_mu2 + deriv_mu2_offset);
						//gauss_sigma_sum_abs = caffe_cpu_asum(kernel_size, deriv_sigma + deriv_sigma_offset);
						// wrong version for normalization with gauss square - but it worked !!
						gauss_mu1_sum = caffe_sum(kernel_size, deriv_mu1 + deriv_mu1_offset);
						gauss_mu2_sum = caffe_sum(kernel_size, deriv_mu2 + deriv_mu2_offset);
						gauss_sigma_sum = caffe_sum(kernel_size, deriv_sigma + deriv_sigma_offset);
					} else {
						gauss_mu1_sum = 2*caffe_cpu_dot(kernel_size, deriv_weight + tmp_deriv_weight_offset, deriv_mu1 + deriv_mu1_offset);
						gauss_mu2_sum = 2*caffe_cpu_dot(kernel_size, deriv_weight + tmp_deriv_weight_offset, deriv_mu2 + deriv_mu2_offset);
						gauss_sigma_sum = 2*caffe_cpu_dot(kernel_size, deriv_weight + tmp_deriv_weight_offset, deriv_sigma + deriv_sigma_offset);
					}

					// add derivatives of normalization factor
					// i.e. G(x)/N(x) from GMM equation has derivate as N(x)*G'(x) - G(x) * N'(x)
					// where  G is guassian and N is sum of gaussian for normalization,
					// while G' is derivative of gaussian and N' is sum of derivative of gaussian
					gauss_mu1_sum = abs(gauss_mu1_sum) > 1e-10 ? gauss_mu1_sum : 0;
					gauss_mu2_sum = abs(gauss_mu2_sum) > 1e-10 ? gauss_mu2_sum : 0;

					caffe_cpu_axpby(kernel_size, -1* gauss_mu1_sum, tmp_deriv_weight + tmp_deriv_weight_offset, norm_factor, deriv_mu1 + deriv_mu1_offset);
					caffe_cpu_axpby(kernel_size, -1* gauss_mu2_sum, tmp_deriv_weight + tmp_deriv_weight_offset, norm_factor, deriv_mu2 + deriv_mu2_offset);
					caffe_cpu_axpby(kernel_size, -1* gauss_sigma_sum, tmp_deriv_weight + tmp_deriv_weight_offset, norm_factor, deriv_sigma + deriv_sigma_offset);


					caffe_scal(kernel_size, (Dtype)1.0/norm_factor, tmp_deriv_weight + tmp_deriv_weight_offset);
					caffe_scal(kernel_size, (Dtype)w/(norm_factor*norm_factor), deriv_mu1 + deriv_mu1_offset);
					caffe_scal(kernel_size, (Dtype)w/(norm_factor*norm_factor), deriv_mu2 + deriv_mu2_offset);
					caffe_scal(kernel_size, (Dtype)w/(norm_factor*norm_factor), deriv_sigma + deriv_sigma_offset);


				}


				// generate seperable kernels
				Dtype w_normed = sqrt(std::abs<Dtype>(w*normalize_factor_inv));

				// since normed weight is a square-root of w*normalize_factor we loose a sign so we need to bring it back in
				// add it only to one horizotnal or vertical one, but NOT BOTH since they will cancel each-other in final equation
				Dtype w_sign = w*normalize_factor_inv < 0 ? -1 : 1;

				if (weight_horiz_buffer != NULL) {
					weights_index = weight_horiz_buffer->offset(i,k,j); //((j*this->conv_out_channels_ + i) * NUM_GAUSS + k) * this->kernel_w_;
					for (int x = 0; x < kernel_w; ++x) {

						Dtype dist_x = x - mu1;
						Dtype dist_x_2 = dist_x*dist_x;


						weight_horiz[weights_index++] = w_normed * w_sign * exp( -dist_x_2 * sigma2_inv_half);
					}
				}
				if (weight_vert_buffer != NULL) {
					weights_index = weight_vert_buffer->offset(i,k,j); //((j*this->conv_in_channels_ + i)* NUM_GAUSS + k) * this->kernel_h_;
					for (int y = 0; y < kernel_h; ++y) {

						Dtype dist_y = y - mu2;
						Dtype dist_y_2 = dist_y*dist_y;

						weight_vert[weights_index++] = w_normed * exp( -dist_y_2 * sigma2_inv_half);
					}
				}

			}
			if (is_backward_pass) {


				// notice: deriv_error_buffer_ has first two dimensions switched to enable more efficent computation of bottom error in backward process
				int deriv_error_offset = deriv_error_buffer->offset(j,i);

				int kernel_size = kernel_h * kernel_w;
				for (int weights_index = 0; weights_index < kernel_size; ++weights_index) {
					// copy weights (GMM) into derivate/error weights which are the same but in reverse order
					deriv_error[deriv_error_offset + kernel_size - weights_index - 1] = weight[weights_offset + weights_index];
				}

			}
		}
	}

	free(weight_tmp);

	clock_t end_t = clock();

	LOG(INFO) << "precompute_guassian_weights (CPU) done in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);
}

#ifdef CPU_ONLY

template <typename Dtype>
void BaseGaussianConvLayer<Dtype>::precompute_guassian_weights_gpu(bool is_backward_pass) {
	// re-direct to CPU version
	precompute_guassian_weights(is_backward_pass);
}

#endif



template <typename Dtype>
void GaussianConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	LOG(INFO) << "called Forward_cpu";
	// forward CPU:
	// go over each bottom sub-feature i
	// go over each top feature j
	// go over K gaussians (K = 4)
	// convolve bottom feature with k-th gaussian 
	// and add values/matrix to j-th result
	// add bias b[j] to  j-th result

	// precompute kernels from gaussian parameters but only for forward pass
	//this->precompute_guassian_weights(false);
	if (this->using_gpu)
		this->precompute_guassian_weights_gpu(true); // avoid second call in Backward_cpu and just compute everything here
	else
		this->precompute_guassian_weights(true); // avoid second call in Backward_cpu and just compute everything here

	const Dtype* weight = this->using_gpu ? this->weight_buffer_->gpu_data() : this->weight_buffer_->cpu_data();
	const Dtype* bias = this->using_gpu ? this->param_buffer_bias_->gpu_data() : this->param_buffer_bias_->cpu_data();

	const Dtype* weight_vert = this->using_gpu ? this->weight_vert_buffer_->gpu_data() : this->weight_vert_buffer_->cpu_data();
	const Dtype* weight_horiz = this->using_gpu ? this->weight_horiz_buffer_->gpu_data() : this->weight_horiz_buffer_->cpu_data();

	const int* is_weight_enabled = this->using_gpu ? this->is_weight_enabled_buffer_.gpu_data() : this->is_weight_enabled_buffer_.cpu_data();

	Dtype* col_buff = this->tmp_buffer_sepe_1_.mutable_cpu_data();
	Dtype* second_col_buff = this->tmp_buffer_sepe_2_.mutable_cpu_data();

	clock_t start_time_all = clock();

	clock_t time_org = 0, time_seperable = 0;

	for (int i = 0; i < bottom.size(); ++i) {

		const Dtype* bottom_data = this->using_gpu ? bottom[i]->gpu_data() : bottom[i]->cpu_data();
		Dtype* top_data = this->using_gpu ? top[i]->mutable_gpu_data() : top[i]->mutable_cpu_data();

//#define PROFILE_FW_PASS_SEPERABLE
//#define PROFILE_FW_PASS_SEPERABLE_DETAIL


#ifdef PROFILE_FW_PASS_SEPERABLE
		Dtype* top_diff = this->using_gpu ? top[i]->mutable_gpu_diff() : top[i]->mutable_cpu_diff();
#endif

		for (int n = 0; n < this->num_; ++n) {
			clock_t time_org = 0, time_seperable = 0;


#ifdef PROFILE_FW_PASS_SEPERABLE
			clock_t start_t = clock();
#endif
			if (this->use_gmm_seperable_kernels)
				this->forward_cpu_gpu_seperable(bottom_data + bottom[i]->offset(n), weight_vert, weight_horiz, is_weight_enabled, top_data + top[i]->offset(n), col_buff, second_col_buff);
			else
				this->forward_cpu_gpu_gemm(bottom_data + bottom[i]->offset(n), weight, top_data + top[i]->offset(n));
#ifdef PROFILE_FW_PASS_SEPERABLE
			time_org += clock() - start_t;

			start_t = clock();
			this->forward_cpu_gpu_seperable(bottom_data + bottom[i]->offset(n), weight_vert, weight_horiz, is_weight_enabled, top_diff + top[i]->offset(n), col_buff, second_col_buff);
			time_seperable += clock() - start_t;

			LOG(INFO) << "all forward_cpu_gpu defaut done in " << (((float)(time_org))/CLOCKS_PER_SEC);
			LOG(INFO) << "all forward_cpu_gpu_seperable done in " << (((float)(time_seperable))/CLOCKS_PER_SEC);
#endif
			if (this->bias_term_) {
				this->forward_cpu_gpu_bias(top_data + top[i]->offset(n), bias);
			}
		}
#ifdef PROFILE_FW_PASS_SEPERABLE
		caffe_sub(top[i]->count(), top_data, top_diff, top_diff);
		Dtype mean_error = caffe_cpu_asum(top[i]->count(), top_diff) /top[i]->count();

		LOG(INFO) << "mean error  " << mean_error;
#endif
	}
	clock_t end_time_all = clock();
	clock_t time_all = end_time_all - start_time_all;

	LOG(INFO) << "finished Forward_cpu";
	LOG(INFO) << "all Forward_cpu done in " << (((float)(time_all))/CLOCKS_PER_SEC);
}


template <typename Dtype>
void GaussianConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	LOG(INFO) << "called Backward_cpu";
	bool do_mean_optmization = this->gmm_mean_iteration_step > 0 && this->current_iteration_index % this->gmm_mean_iteration_step == 0 ? true : false;
	bool do_sigma_optmization = this->gmm_sigma_iteration_step > 0 && this->current_iteration_index % this->gmm_sigma_iteration_step == 0 ? true : false;

	this->current_iteration_index++;

	// backward CPU:
	{
		for (int i = 0; i < this->blobs_.size(); ++i) {
			if (this->using_gpu)
				caffe_gpu_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_gpu_diff());
			else
				caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_diff());
		}
	}
	//int NUM_GAUSS_PER_AXIS = this->layer_param_.convolution_param().number_gauss();
	//int NUM_GAUSS =  NUM_GAUSS_PER_AXIS * NUM_GAUSS_PER_AXIS;

	// precompute kernels from gaussian parameters for forward and backward pass
	//this->precompute_guassian_weights(true);


	Blob<Dtype>& gauss_param_buffer_w = *this->param_buffer_w_;
	Blob<Dtype>& gauss_param_buffer_mu1 = *this->param_buffer_mu1_;
	Blob<Dtype>& gauss_param_buffer_mu2 = *this->param_buffer_mu2_;
	Blob<Dtype>& gauss_param_buffer_sigma = *this->param_buffer_sigma_;

	const Dtype* weight = this->using_gpu ? this->weight_buffer_->gpu_data() :  this->weight_buffer_->cpu_data();

	// clear values for gauassian parameters diffs
	if (this->param_propagate_down_[0]) {
		if (this->using_gpu) {

			caffe_gpu_set(gauss_param_buffer_w.count(), Dtype(0), gauss_param_buffer_w.mutable_gpu_diff());
			caffe_gpu_set(gauss_param_buffer_mu1.count(), Dtype(0), gauss_param_buffer_mu1.mutable_gpu_diff());
			caffe_gpu_set(gauss_param_buffer_mu2.count(), Dtype(0), gauss_param_buffer_mu2.mutable_gpu_diff());
			caffe_gpu_set(gauss_param_buffer_sigma.count(), Dtype(0), gauss_param_buffer_sigma.mutable_gpu_diff());

		} else {
			caffe_set(gauss_param_buffer_w.count(), Dtype(0), gauss_param_buffer_w.mutable_cpu_diff());
			caffe_set(gauss_param_buffer_mu1.count(), Dtype(0), gauss_param_buffer_mu1.mutable_cpu_diff());
			caffe_set(gauss_param_buffer_mu2.count(), Dtype(0), gauss_param_buffer_mu2.mutable_cpu_diff());
			caffe_set(gauss_param_buffer_sigma.count(), Dtype(0), gauss_param_buffer_sigma.mutable_cpu_diff());
		}
	}

	//cudaDeviceSynchronize();

	clock_t deriv_conv_time = 0;
	clock_t im2col_time = 0;
	clock_t acum_bottom_time = 0;
	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = this->using_gpu ? top[i]->gpu_diff() : top[i]->cpu_diff();
		const Dtype* bottom_data = this->using_gpu ? bottom[i]->gpu_data() : bottom[i]->cpu_data();

		Dtype* bottom_diff = this->using_gpu ? bottom[i]->mutable_gpu_diff() : bottom[i]->mutable_cpu_diff();


		// Bias gradient, if necessary.
		if (this->bias_term_ && this->param_propagate_down_[1]) {
			Dtype* bias_diff = this->using_gpu ? this->param_buffer_bias_->mutable_gpu_diff() : this->param_buffer_bias_->mutable_cpu_diff();
			for (int n = 0; n < this->num_; ++n) {
				this->backward_cpu_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
			}
		}

		if (propagate_down[i]) {
			for (int n = 0; n < this->num_; ++n) {

				// calculate error matrices for lower layer
				// go over each top feature j
				// go over K gaussians (K = 4)
				// convolve top j-th error matrix with k-th gaussian
				// multiply result with bottom activation values (element wise)
				// save result to top_error[i,k]

				this->backward_cpu_gpu_gemm(top_diff + top[i]->offset(n), weight, bottom_diff + bottom[i]->offset(n));

			}
		}
		if (this->param_propagate_down_[0]) {
			clock_t start_t, end_t;
			/*
			Dtype* accumulated_bottom_data = accum_bottom_.mutable_gpu_data();

			cudaDeviceSynchronize();

			start_t = clock();
			//caffe_gpu_add_elementwise(accum_bottom_.count(), bottom_data, accumulated_bottom_data, this->num_);

			cudaDeviceSynchronize();

			end_t = clock();
			acum_bottom_time += end_t - start_t;

			//bottom_data = (const Dtype*)accumulated_bottom_data;
			*/
			for (int n = 0; n < this->num_; ++n) {
			//for (int n = 0; n < 1; ++n) {
				// calculate gradients for each gaussain parameter (weights, mean [x,y], sigma)
				// go over each top feature j
				// go over K gaussians (K = 4)
				// convolve top j-th error matrix with k-th gaussian
				// multiply result with bottom activation values (element wise)
				// save result to top_error[i,k]

				// convert bottom activation data into column array where convolution can be performed using matrix multiplication
				start_t = clock();
				if (this->using_gpu)
					im2col_gpu(bottom_data + bottom[i]->offset(n),
							this->conv_in_channels_,
							this->conv_in_height_, this->conv_in_width_,
							this->kernel_h_, this->kernel_w_,
							this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_,
							1,1,
							this->col_buffer_.mutable_gpu_data());
				else
					im2col_cpu(bottom_data + bottom[i]->offset(n),
							this->conv_in_channels_,
							this->conv_in_height_, this->conv_in_width_,
							this->kernel_h_, this->kernel_w_,
							this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_,
							1,1,
							this->col_buffer_.mutable_cpu_data());
				end_t = clock();

				im2col_time +=  end_t-start_t;

				//cudaDeviceSynchronize();

				start_t = clock();
				// compute convolution with bottom activation data (in column format) with kernels for weight derivative,
				// do dot-product of resut with top error and store final result to gaussian parameter diffs (i.e to diff of blobs_[0])
				compute_parameter_deriv(n, this->col_buffer_, *this->deriv_weight_buffer_, *top[i], gauss_param_buffer_w, 0);

				// do the same for means (mu1, mu1) and sigma
				if (do_mean_optmization) {
					compute_parameter_deriv(n, this->col_buffer_, *this->deriv_mu1_buffer_, *top[i], gauss_param_buffer_mu1, 0);
					compute_parameter_deriv(n, this->col_buffer_, *this->deriv_mu2_buffer_, *top[i], gauss_param_buffer_mu2, 0);
				}
				if (do_sigma_optmization)
					compute_parameter_deriv(n, this->col_buffer_, *this->deriv_sigma_buffer_, *top[i], gauss_param_buffer_sigma, 0);

				end_t = clock();

				deriv_conv_time +=  end_t-start_t;
			}
		}
	}
	clock_t start_t = clock();
	//cudaDeviceSynchronize();
	//const Dtype* param_w_buffer__ = gauss_param_buffer_w.cpu_data();

	clock_t end_t = clock();
	deriv_conv_time +=  end_t-start_t;

	if (this->use_gmm_weight_normalization) {
		LOG(INFO) << "doing gmm weight normalization";

		CHECK_EQ(0,1) << "GMM weight normalization not implemented with new version!!" ;

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
				w_sum += caffe_cpu_asum(this->NUM_GAUSS, param_w_buffer + gauss_param_buffer_w.offset(0,s,f)); // TODO: fix S and F indexes

				// sum gradients of all subfeatures from the same feature weighted by individual GMM weights
				weighted_gradient_sum += caffe_cpu_dot(this->NUM_GAUSS,
															param_w_buffer + gauss_param_buffer_w.offset(0,s,f),
															param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f)); // TODO: fix S and F indexes
			}

			// normalize by sum of weights
			weighted_gradient_sum = weighted_gradient_sum / w_sum;


			for (int s = 0; s < this->conv_in_channels_; ++s) {

				// from abs
				caffe_mul(this->NUM_GAUSS, param_w_sign + this->tmp_w_sign_.offset(0,s,f), // TODO: fix S and F indexes
									param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f),
									param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f));

				// subtract normalized weighted gradient sum from each gradient
				caffe_add_scalar(this->NUM_GAUSS, -1 * weighted_gradient_sum, param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f)); // TODO: fix S and F indexes

				// them multiply by sign(w)
				caffe_mul(this->NUM_GAUSS, param_w_sign + this->tmp_w_sign_.offset(0,s,f), // TODO: fix S and F indexes
							param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f),
							param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f));

				// and finally add normalization factor
				caffe_scal(this->NUM_GAUSS, (Dtype)1./w_sum, param_w_buffer_diff + gauss_param_buffer_w.offset(0,s,f)); // TODO: fix S and F indexes

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
	//Dtype norm_factor = 1.0 / (Dtype)this->conv_out_spatial_dim_; // THIS SHOULD BE CORRECT ONE !!!
	Dtype norm_factor = 1.0;

	if (this->using_gpu) {

		Dtype* gauss_w_deriv = gauss_param_buffer_w.mutable_gpu_diff();
		Dtype* gauss_mu1_deriv = gauss_param_buffer_mu1.mutable_gpu_diff();
		Dtype* gauss_mu2_deriv = gauss_param_buffer_mu2.mutable_gpu_diff();
		Dtype* gauss_sigma_deriv = gauss_param_buffer_sigma.mutable_gpu_diff();

		// add default normalizing factor to all parameters
		caffe_gpu_scal(gauss_param_buffer_w.count(), norm_factor, gauss_w_deriv);
		caffe_gpu_scal(gauss_param_buffer_mu1.count(), norm_factor, gauss_mu1_deriv);
		caffe_gpu_scal(gauss_param_buffer_mu2.count(), norm_factor, gauss_mu2_deriv);
		caffe_gpu_scal(gauss_param_buffer_sigma.count(), norm_factor, gauss_sigma_deriv);

	} else {
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


	LOG(INFO) << "size of tmp_buffer " << this->tmp_buffer_.count() << " = " << this->tmp_buffer_.shape(0) << " x " << this->tmp_buffer_.shape(1) << " x " << this->tmp_buffer_.shape(2) << " x " << this->tmp_buffer_.shape(3);
	LOG(INFO) << "size of tmp_buffer " << this->tmp_buffer_.count() << " = " << this->tmp_buffer_.shape(0) * this->tmp_buffer_.shape(1) * this->tmp_buffer_.shape(2) << " x " << this->tmp_buffer_.shape(3);
	LOG(INFO) << "finished Backward_cpu";
	LOG(INFO) << "all acum_bottom done in " << (((float)(acum_bottom_time))/CLOCKS_PER_SEC);
	LOG(INFO) << "all im2col done in " << (((float)(im2col_time))/CLOCKS_PER_SEC);
	LOG(INFO) << "all compute_parameter_deriv done in " << (((float)(deriv_conv_time))/CLOCKS_PER_SEC);
}
template <typename Dtype>
void GaussianConvLayer<Dtype>::compute_parameter_deriv(int num_iter, 
		const Blob<Dtype>& col_activations_buffer, const Blob<Dtype>& deriv_kernels_buffer,
		//const Blob<Dtype>& top_error_buffer,
		Blob<Dtype>& top_error_buffer,
		Blob<Dtype>& param_output_buffer, int param_output_offset) {

//	clock_t start_t = clock();

	//int NUM_GAUSS_PER_AXIS = this->layer_param_.convolution_param().number_gauss();
	//int NUM_GAUSS =  NUM_GAUSS_PER_AXIS * NUM_GAUSS_PER_AXIS;

	const Dtype* col_buff = this->using_gpu ? col_activations_buffer.gpu_data() : col_activations_buffer.cpu_data();
	const Dtype* deriv_kernels = this->using_gpu ? deriv_kernels_buffer.gpu_data() : deriv_kernels_buffer.cpu_data();// around 0.05 s for copying to GPU per layer for 16-feature, 16-subfeture of size 5x5 kernels
	Dtype* top_error = this->using_gpu ? (Dtype*)top_error_buffer.gpu_diff() : (Dtype*)top_error_buffer.cpu_diff(); // using diff !!

	Dtype* param_output = this->using_gpu ? param_output_buffer.mutable_gpu_diff() : param_output_buffer.mutable_cpu_diff(); // using diff !!

	Dtype* tmp_buff_all = this->using_gpu ? tmp_buffer_.mutable_gpu_data() : tmp_buffer_.mutable_cpu_data();

	for (int s = 0; s < this->conv_in_channels_; ++s) {

		Dtype* tmp_buff = tmp_buff_all + tmp_buffer_.offset(s);

		// compute convolution of activations with weight deriv gaussian for all s-th sub-feature (i.e. NUM_GAUSS * NUM_OUT_CHANNELS number of convolutions)
		A[s] = deriv_kernels + deriv_kernels_buffer.offset(s);
		B[s] = col_buff + col_activations_buffer.offset(s);
		C[s] = tmp_buff;
	}
#ifndef CPU_ONLY
	if (this->using_gpu) {
		cudaMemcpy(d_A, A, this->conv_in_channels_ *sizeof(Dtype*), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, this->conv_in_channels_ *sizeof(Dtype*), cudaMemcpyHostToDevice);
		cudaMemcpy(d_C, C, this->conv_in_channels_ *sizeof(Dtype*), cudaMemcpyHostToDevice);
	} else
#endif
	{
		d_A = (Dtype**)A;
		d_B = (Dtype**)B;
		d_C = C;
	}

	caffe_cpu_gpu_gemm_batched(CblasNoTrans, CblasNoTrans,
					this->conv_out_channels_ * this->NUM_GAUSS,
					this->conv_out_spatial_dim_,
					this->kernel_h_ * this->kernel_w_,
					(Dtype)1., (const Dtype**)d_A, (const Dtype**)d_B,
					(Dtype)0., d_C, this->conv_in_channels_);

	//int tmp_buffer_count = 50;
	//int batch_mul_size = 10;
	int tmp_buffer_count = tmp_buffer_.count();
	int batch_mul_size = this->num_output_ * this->height_out_ * this->width_out_;
	int batch_sum_size = this->height_out_ * this->width_out_;
	int size_params = this->conv_in_channels_ * this->conv_out_channels_ * this->NUM_GAUSS;

	//Blob<Dtype> top_error_buffer_(top_error_buffer.shape());
	Blob<Dtype>& top_error_buffer_ = top_error_buffer;

//#define CHECK_GPU_SUM_MUL

#ifdef CHECK_GPU_SUM_MUL
	if (0){
		LOG(INFO) << "copied to cpu for setting";
		Dtype* ptr_cpu_error = top_error_buffer_.mutable_cpu_diff() + top_error_buffer.offset(num_iter);
		Dtype* ptr_cpu_gpu = tmp_buffer_.mutable_cpu_data();

		for (int i = 0 ; i< tmp_buffer_count; ++i)
			ptr_cpu_gpu[i] = i*0.0000003213;

		for (int i = 0 ; i< batch_mul_size; ++i)
			ptr_cpu_error[i] = i*0.000000000213;

		ptr_cpu_gpu = tmp_buffer_.mutable_gpu_data();
	}

	top_error = this->using_gpu ? (Dtype*)top_error_buffer.gpu_diff() : (Dtype*)top_error_buffer.cpu_diff(); // using diff !!
	param_output = this->using_gpu ? param_output_buffer.mutable_gpu_diff() : param_output_buffer.mutable_cpu_diff(); // using diff !!
	tmp_buff_all = this->using_gpu ? tmp_buffer_.mutable_gpu_data() : tmp_buffer_.mutable_cpu_data();


	caffe_gpu_memcpy(param_output_buffer.count()*sizeof(Dtype),param_output, tmp_blob_.mutable_gpu_diff());
//	caffe_gpu_set(param_output_buffer.count(), (Dtype)0, tmp_blob_.mutable_gpu_diff());

	for (int g = 0; g < this->NUM_GAUSS; ++g) {
		for (int s = 0; s < this->conv_in_channels_; ++s) {

			//const Dtype* tmp_buff = this->using_gpu ? tmp_buffer_[s]->gpu_data() : tmp_buffer_[s]->cpu_data();

			// perform dot-product of each convolution result with top error diff for each feature individualy

			for (int f = 0; f < this->conv_out_channels_; ++f) {
				A[s * this->conv_out_channels_ + f] = tmp_buff_all + tmp_buffer_.offset(s,g,f);//,f);
				B[s * this->conv_out_channels_ + f] = top_error + top_error_buffer.offset(num_iter,f);
				//C[s * this->conv_out_channels_ + f] = param_output + param_output_buffer.offset(param_output_offset, s, g, f);
				C[s * this->conv_out_channels_ + f] = tmp_blob_.mutable_gpu_diff() + param_output_buffer.offset(param_output_offset, s, g, f);

			}
		}
	#ifndef CPU_ONLY
		if (this->using_gpu) {
			cudaMemcpy(d_A, A, this->conv_out_channels_* this->conv_in_channels_ *sizeof(Dtype*), cudaMemcpyHostToDevice);
			cudaMemcpy(d_B, B, this->conv_out_channels_* this->conv_in_channels_ *sizeof(Dtype*), cudaMemcpyHostToDevice);
			cudaMemcpy(d_C, C, this->conv_out_channels_* this->conv_in_channels_ *sizeof(Dtype*), cudaMemcpyHostToDevice);
		} else
	#endif
		{
			d_A = (Dtype**)A;
			d_B = (Dtype**)B;
			d_C = C;
		}

		caffe_cpu_gpu_gemm_batched(CblasNoTrans, CblasNoTrans,
							1,
							1,
							this->conv_out_spatial_dim_,
							(Dtype)1., (const Dtype**)d_A, (const Dtype**)d_B,
							(Dtype)1., d_C, this->conv_out_channels_* this->conv_in_channels_);


	}

	if (1) {
		cudaDeviceSynchronize();

		int g = 0;
		int s = 0;
		int f = 0;
		const Dtype* ptr_cpu_tmp = tmp_buffer_.cpu_data()  + tmp_buffer_.offset(s,g,f);
		const Dtype* ptr_cpu_error = top_error_buffer.cpu_diff()  + top_error_buffer.offset(num_iter,f);
		const Dtype* ptr_cpu_param = tmp_blob_.cpu_diff()  + param_output_buffer.offset(param_output_offset,s,g,f);

		float sum_value = 0;
		int batch_offset = 0;
		for (int j = 0; j < batch_sum_size; ++j) {
//			LOG(INFO)  << "\t actual table values: tmp " << ptr_cpu_tmp[j + batch_offset] << " error " << ptr_cpu_error[j + batch_offset];
			sum_value += ptr_cpu_tmp[j + batch_offset]* ptr_cpu_error[j + batch_offset];
		}
		LOG(INFO)  << "-> from table s: " << s << " g: " << f << " f: " << f << " where GPU dot-product is " << ptr_cpu_param[0] << " and cpu dot-product is " << sum_value;

	}
	// we ensured resultes are stacked with outer index S, middle index G and inner most index F (i.e. S x G x F x [H*W] with last index changing fastest along the row)
	// and we can now reshape this memory so that each row contains results from different F (i.e. into [S * G] x [F*H*W]
	// so that now each row can be simply multipled by back-propagated error for different F-s and then summed for each individual W*H
	// to get dot-product with back-propagation error

	bool dump_data = 0 && num_iter == 1 && tmp_buffer_.num() == 16 && tmp_buffer_.channels() == 4 && tmp_buffer_.height() == 16 && tmp_buffer_.width() == 11844;


	if (dump_data) {

		cv::Mat param_output_mat(1,this->conv_out_channels_ * this->conv_in_channels_ * this->NUM_GAUSS, CV_32FC1, (void*)(tmp_blob_.mutable_cpu_diff() + param_output_buffer.offset(param_output_offset)));

		std::ofstream param_output_file("/home/domen/Downloads/1/param_output_mat_correct.m");
		param_output_file << "param_output_mat_correct_ = " << param_output_mat << ";";
		param_output_file.close();

		cv::Mat tmp_buffer_mat(1, tmp_buffer_.count(), CV_32FC1, (void*)tmp_buffer_.mutable_cpu_data());
		cv::Mat top_error_mat(1, this->num_output_ * this->height_out_ * this->width_out_, CV_32FC1, (void*)(top_error_buffer.cpu_diff() + top_error_buffer.offset(num_iter)));

		LOG(INFO) << "size of tmp_buffer_: " << tmp_buffer_.num() << " x " << tmp_buffer_.channels() << " x " << tmp_buffer_.height() << " x " << tmp_buffer_.width() ;
		std::ofstream tmp_buffer_file("/home/domen/Downloads/1/tmp_buffer_mat.m");
		tmp_buffer_file << "tmp_buffer_ = " << tmp_buffer_mat << ";";
		tmp_buffer_file.close();

		LOG(INFO) << "size of top_error_buffer: " <<  top_error_buffer.num() << " x " << top_error_buffer.channels() << " x " << top_error_buffer.height() << " x " << top_error_buffer.width();
		std::ofstream top_error_file("/home/domen/Downloads/1/top_error_mat.m");
		top_error_file << "top_error_mat_ = " << top_error_mat << ";";
		top_error_file.close();

	}

	cudaDeviceSynchronize();



	Blob<Dtype> cpu_tmp_buffer_org(tmp_buffer_.shape());
	cpu_tmp_buffer_org.CopyFrom(tmp_buffer_,false, true);

	Blob<Dtype> cpu_tmp_buffer_(tmp_buffer_.shape());
	cpu_tmp_buffer_.CopyFrom(tmp_buffer_,false, true);

	caffe_cpu_mul_batch(tmp_buffer_count, cpu_tmp_buffer_.mutable_cpu_data(), (Dtype*)top_error_buffer_.mutable_cpu_diff() + top_error_buffer.offset(num_iter),  cpu_tmp_buffer_.mutable_cpu_data(), batch_mul_size);

	tmp_buff_all = tmp_buffer_.mutable_gpu_data();

	(Dtype*)top_error_buffer_.mutable_gpu_diff() + top_error_buffer.offset(num_iter);
	//LOG(INFO) << "running caffe_gpu_mul";
	// we use "batched" multiply to replicate multiple of each row with the back-propagation vector
	clock_t start_t = clock();
#endif

	caffe_gpu_mul_batched(tmp_buffer_count, tmp_buff_all, top_error + top_error_buffer.offset(num_iter),  tmp_buff_all, batch_mul_size);
	//caffe_gpu_mul(1024, tmp_buff_all, tmp_buff_all,  tmp_buff_all);
#ifdef CHECK_GPU_SUM_MUL
	clock_t end_t = clock();
	LOG(INFO) << "caffe_gpu_mul done in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);
	//LOG(INFO) << "done caffe_gpu_mul";

	cudaDeviceSynchronize();

	Dtype* ptr_cpu_error = top_error_buffer_.mutable_cpu_diff();
	Dtype* ptr_cpu_org = cpu_tmp_buffer_org.mutable_cpu_data();
	Dtype* ptr_cpu_gpu = tmp_buffer_.mutable_cpu_data();
	Dtype* ptr_cpu = cpu_tmp_buffer_.mutable_cpu_data();

	int count_mismatch = 0;
	for (int i = 0; i < tmp_buffer_count; ++i) {
		if (std::abs<Dtype>(ptr_cpu_gpu[i] - ptr_cpu[i]) > 0.0000001)
		{
		//if (ptr_cpu_gpu[i] != 0  || ptr_cpu[i] != 0)
			LOG(INFO) << "index " << i << " gpu: " << ptr_cpu_gpu[i] << " vs cpu: " <<  ptr_cpu[i] << " from tmp: " <<  ptr_cpu_org[i] << " and error " << ptr_cpu_error[i % batch_mul_size];
			count_mismatch++;
		}
	}

	LOG(INFO) << "number of mismatch between cpu and gpu in MUL is " << count_mismatch;

	//tmp_buff_all = tmp_buffer_.mutable_gpu_data();

	if (dump_data) {
		cudaDeviceSynchronize();

		cv::Mat tmp_buffer_midle_mat(1,tmp_buffer_.count(), CV_32FC1, (void*)tmp_buffer_.mutable_cpu_data());

		std::ofstream tmp_buffer_middle_file("/home/domen/Downloads/1/tmp_buffer_midle_mat.m");
		tmp_buffer_middle_file << "tmp_buffer_midle_mat_ = " << tmp_buffer_midle_mat << ";";
		tmp_buffer_middle_file.close();

	}
	LOG(INFO) << "evaluating caffe_gpu_sum";
	Dtype* tmp_blob_ptr = tmp_blob_.mutable_gpu_data();


	//size_params = 5;
	//tmp_buffer_count = 50;
	//batch_sum_size = 10;


	tmp_buff_all = tmp_buffer_.mutable_gpu_data();
	param_output = param_output_buffer.mutable_gpu_diff() + param_output_buffer.offset(param_output_offset);
	// then we use segmented reduce-sum from NVIDIA cub library to sum each row in [S * G * F] x [H*W] thus producing [S x G x F] output
	start_t = clock();

//	caffe_cpu_sum(tmp_buffer_count, cpu_tmp_buffer_.mutable_cpu_data(), tmp_blob_.mutable_cpu_diff() + param_output_buffer.offset(param_output_offset), batch_sum_size); // NOTE: this might require with_add=true, but not sure
#endif
	caffe_gpu_sum(tmp_buffer_count, tmp_buff_all, param_output, size_params, this->tmp_index_gpu, true, 0);
	//caffe_gpu_sum(tmp_buffer_count, tmp_buff_all, param_output, batch_sum_size); // NOTE: this might require with_add=true, but not sure
#ifdef CHECK_GPU_SUM_MUL
	//caffe_gpu_sum(tmp_buffer_.count(), tmp_buff_all, tmp_blob_ptr, this->conv_out_channels_ * this->conv_in_channels_ * this->NUM_GAUSS) // NOTE: this might require with_add=true, but not sure;
	//caffe_gpu_add(this->conv_out_channels_ * this->conv_in_channels_ * this->NUM_GAUSS, tmp_blob_ptr, param_output + param_output_buffer.offset(param_output_offset), param_output + param_output_buffer.offset(param_output_offset));

	end_t = clock();
	LOG(INFO) << "caffe_gpu_sum done in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);

	Dtype* ptr_gpu_tmp = tmp_buffer_.mutable_cpu_data();
	Dtype* ptr_cpu_tmp = cpu_tmp_buffer_.mutable_cpu_data();

	ptr_cpu_gpu = param_output_buffer.mutable_cpu_diff()  + param_output_buffer.offset(param_output_offset);
	ptr_cpu = tmp_blob_.mutable_cpu_diff() + param_output_buffer.offset(param_output_offset);


//	size_params = 1;
	count_mismatch = 0;
	for (int i = 0; i < size_params; ++i) {
		if (ptr_cpu[i] == 0 || ptr_cpu_gpu[i] == 0 || std::abs<Dtype>((ptr_cpu_gpu[i] - ptr_cpu[i])/ptr_cpu[i]) > 0.01  )
		{
			count_mismatch++;


			if (ptr_cpu[i] == 0 || ptr_cpu_gpu[i] == 0 ) continue;
			int batch_offset = i * batch_sum_size;
			for (int j = 0; j < batch_sum_size; ++j)
				if (std::abs<Dtype>((ptr_cpu_tmp[j + batch_offset] - ptr_gpu_tmp[j + batch_offset])/ptr_cpu_tmp[j + batch_offset]) > 0.000001)
				LOG(INFO)  << "\t actual table values: cpu " << ptr_cpu_tmp[j + batch_offset] << " gpu " << ptr_gpu_tmp[j + batch_offset];

			LOG(INFO) << "-> from index " << i << " cpu: " << ptr_cpu[i] << " vs gpu: " <<  ptr_cpu_gpu[i] << "";


		}
	}

	LOG(INFO) << "number of mismatch between cpu and gpu in SUM is " << count_mismatch;

	//LOG(INFO) << "done caffe_gpu_sum";
	if (dump_data) {
		cv::Mat param_output_mat(1,this->conv_out_channels_ * this->conv_in_channels_ * this->NUM_GAUSS, CV_32FC1, (void*)(param_output_buffer.cpu_diff() + param_output_buffer.offset(param_output_offset)));

		std::ofstream param_output_file("/home/domen/Downloads/1/param_output_mat.m");
		param_output_file << "param_output_mat_ = " << param_output_mat << ";";
		param_output_file.close();


		exit(0);
	}
//	if (num_iter == 1) exit(0);
#endif
//	clock_t end_t = clock();
//	LOG(INFO) << "compute_parameter_deriv done in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC);
}


template <typename Dtype>
void GaussianConvLayer<Dtype>::forward_cpu_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col) {
	if (this->using_gpu)
		this->forward_gpu_gemm(input, weights, output, skip_im2col);
	else
		this->forward_cpu_gemm(input, weights, output, skip_im2col);
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::forward_cpu_gpu_bias(Dtype* output, const Dtype* bias) {
	if (this->using_gpu)
		this->forward_gpu_bias(output, bias);
	else
		this->forward_cpu_bias(output, bias);
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::backward_cpu_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input) {
	if (this->using_gpu)
		this->backward_gpu_gemm(output, weights, input);
	else
		this->backward_cpu_gemm(output, weights, input);
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::weight_cpu_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
	if (this->using_gpu)
		this->weight_gpu_gemm(input, output, weights);
	else
		this->weight_cpu_gemm(input, output, weights);
}

template <typename Dtype>
void GaussianConvLayer<Dtype>::backward_cpu_gpu_bias(Dtype* bias, const Dtype* input) {
	if (this->using_gpu)
		this->backward_gpu_bias(bias, input);
	else
		this->backward_cpu_bias(bias, input);
}


template <typename Dtype>
void GaussianConvLayer<Dtype>::caffe_cpu_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C) {
	if (this->using_gpu)
		caffe_gpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, A, B, beta, C);
	else
		caffe_cpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, A, B, beta, C);

}

template <typename Dtype>
void GaussianConvLayer<Dtype>::caffe_cpu_gpu_gemm_batched(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const Dtype alpha, const Dtype** A, const Dtype** B, const Dtype beta, Dtype** C, int batch_count) {
	if (this->using_gpu) {
		caffe_gpu_gemm_batched<Dtype>(TransA, TransB, M, N, K, alpha, A, B, beta, C, batch_count);
	} else {
		for (int i = 0; i < batch_count; ++i)
			caffe_cpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, A[i], B[i], beta, C[i]);
	}

}

template <typename Dtype>
void GaussianConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	this->using_gpu = true;
	Forward_cpu(bottom, top);
	this->using_gpu = false;
}
template <typename Dtype>
void GaussianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	this->using_gpu = true;
	Backward_cpu(top, propagate_down, bottom);
	this->using_gpu = false;
}

template <typename Dtype>
void convolution_1D(const Dtype* input, const Dtype* weights, const int* is_weight_enabled, Dtype* output, int dim, int num_samples, int num_repeat_samples, int num_group_output, int width, int height, int kernel_size) {
	// input = [(FxGxS)x(HxW)]
	// output = [(FxGxS)x(HxW)]

	const int kernel_half_size = kernel_size / 2;
	const int sample_data_size = width * height;


	const int unpadded_width = width - 2*kernel_half_size;
	const int unpadded_height = height - 2*kernel_half_size;

	// copy only with padded width, but padding in height is not needed
	const int copy_size = width * unpadded_height - 2*kernel_half_size; // since src/dst are aligned to unpadded we need to remove last two padded blocks to avoid out-of-bound copy

	Dtype* output_i = output;
	Dtype* input_i = (Dtype*)input;
	Dtype* kernel_i = (Dtype*)weights + kernel_half_size;
	int* is_weight_enabled_i = (int*)is_weight_enabled;

	// add initial padded border to input_i and output_i
	output_i += width * kernel_half_size + kernel_half_size;

	// if output of several seqeuntial convolutions are summed element-wise then
	// the loop before samples (i.e., g-loop) needs to loop over each group
	// while samples loop needs needs to reduce the number of samples it goes over
	// (num_samples MUST be devisable by num_group_output must)
	num_samples = num_samples / num_group_output;

	const int output_start_offset = dim == 0 ? 1 : width;

	for (int j = 0; j < num_repeat_samples; ++j) {
		Dtype* input_i = (Dtype*)input;

		input_i += width * kernel_half_size + kernel_half_size;

		for (int g = 0; g < num_group_output; g++) {
			for (int i = 0; i < num_samples; ++i) {

				if (*is_weight_enabled_i != 0) {
					for (int k = -kernel_half_size; k <= kernel_half_size; ++k) {

						Dtype* output_i_k = (Dtype*)output_i - k * output_start_offset;
						const Dtype kernel_value = kernel_i[k]; // kernel_i should point to center of pointer by default

						// multiply input with kernel value at k and save it to output shifted by k
						caffe_axpy(copy_size, kernel_value, input_i, output_i_k);
					}
				}
				input_i += sample_data_size;
				kernel_i += kernel_size;

				++is_weight_enabled_i;
			}
			output_i += sample_data_size;
		}
	}
}
template <typename Dtype>
void add_image_padding(const Dtype* image, int num_images, int width, int height, int pad_w, int pad_h, Dtype* output) {
	Dtype* src_ptr = (Dtype*)image;
	Dtype* dst_ptr = (Dtype*)output;

	// border offset on both sides
	const int border_x_offset = 2*pad_w;
	const int border_y_offset = (2 * pad_w + width) * pad_h * 2;

	// first offset for dst should be only half of calculated border offsets
	dst_ptr += border_x_offset/2 + border_y_offset/2;

	for (int i = 0; i < num_images; ++i) {
		for (int y = 0; y < height; ++y) {
			memcpy(dst_ptr, src_ptr, sizeof(Dtype) * width);
			dst_ptr += width + border_x_offset;
			src_ptr += width;
		}
		dst_ptr += border_y_offset;
	}
}

template <typename Dtype>
void remove_image_padding(const Dtype* image, int num_images, int width, int height, int pad_w, int pad_h, Dtype* output) {
	Dtype* src_ptr = (Dtype*)image;
	Dtype* dst_ptr = (Dtype*)output;

	// border offset on both sides
	const int border_x_offset = 2*pad_w;
	const int border_y_offset = (2 * pad_w + width) * pad_h * 2;

	// first offset for src should be only half of calculated border offsets
	src_ptr += border_x_offset/2 + border_y_offset/2;

	for (int i = 0; i < num_images; ++i) {
		for (int y = 0; y < height; ++y) {
			memcpy(dst_ptr, src_ptr, sizeof(Dtype) * width);
			dst_ptr += width;
			src_ptr += width+border_x_offset;
		}
		src_ptr += border_y_offset;
	}
}
template <typename Dtype>
void GaussianConvLayer<Dtype>::forward_cpu_gpu_seperable(const Dtype* input, const Dtype* weights_vert, const Dtype* weights_horiz, const int* is_weight_enabled, Dtype* output, Dtype* col_buff, Dtype* second_col_buff) {
	if (this->using_gpu) {
		CHECK_EQ(0,1) << "Forward-pass with seperable kernel on GPU is not implemented!!";
	} else {

		int width_padded = this->width_ + this->pad_w_*2;
		int height_padded = this->height_ + this->pad_h_*2;

#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		clock_t start_t = clock();
#endif
		// set zero memory for input buffer in first convolution
		caffe_set(width_padded * height_padded * this->conv_in_channels_, (Dtype)0, col_buff);
#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		clock_t zero_first = clock() - start_t;
#endif
		//this->tmp_buffer_sepe_1_.Reshape(1, this->conv_in_channels_, this->height_ + 2 * this->pad_h_, this->width_ + 2 * this->pad_w_);
		//this->tmp_buffer_sepe_2_.Reshape(this->conv_in_channels_, this->conv_out_channels_ * this->NUM_GAUSS, this->height_ + 2 * this->pad_h_, this->width_ + 2 * this->pad_w_);

		//col_buff = this->tmp_buffer_sepe_1_.mutable_cpu_data();
		//second_col_buff = this->tmp_buffer_sepe_2_.mutable_cpu_data();
#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		start_t = clock();
#endif
		// add padding to image and copy it to col_buff
		add_image_padding(input, this->conv_in_channels_, this->width_, this->height_, this->pad_w_, this->pad_h_, col_buff);
#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		clock_t add_padding = clock() - start_t;

		start_t = clock();
#endif
		// clear output memory for the first convolution
		caffe_set(width_padded * height_padded * this->conv_in_channels_ * this->conv_out_channels_ * this->NUM_GAUSS, (Dtype)0, second_col_buff);
#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		clock_t zero_second = clock() - start_t;

		start_t = clock();
#endif
		// perform first convolution with vertical kernel
		convolution_1D(col_buff, weights_vert, is_weight_enabled, second_col_buff, 1, this->conv_in_channels_, this->conv_out_channels_* this->NUM_GAUSS, this->conv_in_channels_, width_padded, height_padded, this->kernel_h_);
#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		clock_t conv1d_first = clock() - start_t;
#endif
		//this->tmp_buffer_sepe_1_.Reshape(1, this->conv_out_channels_, this->height_ + 2 * this->pad_h_, this->width_ + 2 * this->pad_w_);
		//col_buff = this->tmp_buffer_sepe_1_.mutable_cpu_data();

#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		start_t = clock();
#endif
		// clear output memory for the second convolution
		caffe_set(width_padded * height_padded * this->conv_out_channels_, (Dtype)0, col_buff);
#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		clock_t zero_third = clock() - start_t;

		start_t = clock();
#endif
		// perform second convolution with horizontal kernel
		convolution_1D(second_col_buff, weights_horiz, is_weight_enabled, col_buff, 0, this->conv_out_channels_ * this->conv_in_channels_ * this->NUM_GAUSS, 1, this->conv_out_channels_, width_padded, height_padded, this->kernel_w_);

#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		clock_t conv1d_second = clock() - start_t;

		start_t = clock();
#endif
		// remove image padding and save it to output buffer
		remove_image_padding(col_buff, this->conv_out_channels_, this->width_, this->height_, this->pad_w_, this->pad_h_, output);
#ifdef PROFILE_FW_PASS_SEPERABLE_DETAIL
		clock_t remove_padding = clock() - start_t;

		LOG(INFO) << "zero_first  done in " << (((float)(zero_first))/CLOCKS_PER_SEC);
		LOG(INFO) << "add_padding done in " << (((float)(add_padding))/CLOCKS_PER_SEC);
		LOG(INFO) << "zero_second done in " << (((float)(zero_second))/CLOCKS_PER_SEC);
		LOG(INFO) << "conv1d_first done in " << (((float)(conv1d_first))/CLOCKS_PER_SEC);
		LOG(INFO) << "zero_third done in " << (((float)(zero_third))/CLOCKS_PER_SEC);
		LOG(INFO) << "conv1d_second done in " << (((float)(conv1d_second))/CLOCKS_PER_SEC);
		LOG(INFO) << "remove_padding done in " << (((float)(remove_padding))/CLOCKS_PER_SEC);
#endif
		//this->tmp_buffer_sepe_1_.Reshape(this->conv_in_channels_, this->NUM_GAUSS* this->conv_out_channels_, this->height_ + 2* this->pad_h_, this->width_ + 2* this->pad_w_);
		//this->tmp_buffer_sepe_2_.Reshape(this->conv_in_channels_, this->NUM_GAUSS* this->conv_out_channels_, this->height_ + 2* this->pad_h_, this->width_ + 2* this->pad_w_);

	}
}

INSTANTIATE_CLASS(GaussianConvLayer);
INSTANTIATE_CLASS(BaseGaussianConvLayer);
//REGISTER_LAYER_CLASS(GaussianConv);

}  // namespace caffe
