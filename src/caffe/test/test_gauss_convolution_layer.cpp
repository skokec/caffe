#include <boost/smart_ptr/shared_ptr.hpp>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/filler.hpp>
#include <caffe/layers/gauss_conv_layer.hpp>
#include <caffe/layers/cudnn_conv_layer.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/test/test_caffe_main.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>

namespace caffe {

template <typename Dtype>
void compare_blobs(Blob<Dtype>& a, Blob<Dtype>& b, bool compare_diff, Dtype eps) {
	Dtype* data_a = compare_diff ? a.mutable_cpu_diff() : a.mutable_cpu_data();
	Dtype* data_b = compare_diff ? b.mutable_cpu_diff() : b.mutable_cpu_data();
	for (int i = 0; i < a.count(); ++i) {
		EXPECT_NEAR(data_a[i], data_b[i], eps);
	}
}

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int kernel_d, pad_d, stride_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
  } else {
    kernel_d = stride_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r;
                      int in_y = y * stride_h - pad_h + p;
                      int in_x = x * stride_w - pad_w + q;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
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
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class GaussConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GaussConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
		blob_bottom_3_(new Blob<Dtype>(1, 1, 4, 4)),
		//blob_bottom_3_(new Blob<Dtype>(2, 3, 32, 48)),
        blob_top_(new Blob<Dtype>()),
		blob_top_2_(new Blob<Dtype>()),
		blob_top_3_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    filler.Fill(this->blob_bottom_3_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~GaussConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_top_;
    delete blob_top_2_;
    delete blob_top_3_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  Blob<Dtype>* const blob_top_3_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GaussConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(GaussConvolutionLayerTest, TestSetup) {
  /*typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new GaussianConvLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  layer.reset(new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);*/
}
/*
TYPED_TEST(GaussConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(7);
  convolution_param->add_stride(1);
  convolution_param->add_pad(3);

  convolution_param->add_number_gauss(2);

  convolution_param->set_num_output(16);

  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_std(0.01);

  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  convolution_param->mutable_sigma_filler()->set_type("constant");
  convolution_param->mutable_sigma_filler()->set_value(0.8);

  convolution_param->set_gmm_component_border_bound(1.5);
  convolution_param->set_gmm_sigma_lower_bound(0.5);

  convolution_param->set_gmm_weight_normalization(false);
  convolution_param->set_gmm_gauss_normalization(true);
  convolution_param->set_gmm_square_gauss_normalization(true);

  shared_ptr<Layer<Dtype> > layer(
      new GaussianConvLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
*/

TYPED_TEST(GaussConvolutionLayerTest, TestKernelPrecompute) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);

  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
	  layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(7);
  convolution_param->add_stride(1);
  convolution_param->add_pad(3);

  convolution_param->add_number_gauss(2);

  convolution_param->set_num_output(32);

  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_std(0.01);

  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  convolution_param->mutable_sigma_filler()->set_type("constant");
  convolution_param->mutable_sigma_filler()->set_value(0.8);

  convolution_param->set_gmm_component_border_bound(1.5);
  convolution_param->set_gmm_sigma_lower_bound(0.5);

  convolution_param->set_gmm_weight_normalization(false);
  convolution_param->set_gmm_gauss_normalization(true);
  for (int gmm_sqrt_norm = 0; gmm_sqrt_norm <= 1; gmm_sqrt_norm++) {
	convolution_param->set_gmm_square_gauss_normalization((bool)gmm_sqrt_norm);

	shared_ptr<GaussianConvLayer<Dtype> > layer(new GaussianConvLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

	// RUN CPU version
	layer->precompute_guassian_weights(true); // run CPU first since it will force buffers to cpu and would cause errors if gpu would be allocated first

	// store output

	Blob<Dtype> weight_cpu, deriv_error_cpu, deriv_weight_cpu, deriv_mu1_cpu, deriv_mu2_cpu, deriv_sigma_cpu;

	weight_cpu.CopyFrom(*layer->weight_buffer_, false, true);

	deriv_error_cpu.CopyFrom(*layer->deriv_error_buffer_, false, true);
	deriv_weight_cpu.CopyFrom(*layer->deriv_weight_buffer_, false, true);
	deriv_mu1_cpu.CopyFrom(*layer->deriv_mu1_buffer_, false, true);
	deriv_mu2_cpu.CopyFrom(*layer->deriv_mu2_buffer_, false, true);
	deriv_sigma_cpu.CopyFrom(*layer->deriv_sigma_buffer_, false, true);

	// RUN CPU version
	layer->precompute_guassian_weights_gpu(true);
	layer->precompute_guassian_weights_gpu(true);
	layer->precompute_guassian_weights_gpu(true);

	// Check both versions
	Dtype* data_gpu;
	Dtype* data_cpu;

	data_cpu = weight_cpu.mutable_cpu_data();
	data_gpu = layer->weight_buffer_->mutable_cpu_data();
	for (int i = 0; i < weight_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_error_cpu.mutable_cpu_data();
	data_gpu = layer->deriv_error_buffer_->mutable_cpu_data();
	for (int i = 0; i < deriv_error_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_weight_cpu.mutable_cpu_data();
	data_gpu = layer->deriv_weight_buffer_->mutable_cpu_data();
	for (int i = 0; i < deriv_weight_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_mu1_cpu.mutable_cpu_data();
	data_gpu = layer->deriv_mu1_buffer_->mutable_cpu_data();
	for (int i = 0; i < deriv_mu1_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_mu2_cpu.mutable_cpu_data();
	data_gpu = layer->deriv_mu2_buffer_->mutable_cpu_data();
	for (int i = 0; i < deriv_mu2_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_sigma_cpu.mutable_cpu_data();
	data_gpu = layer->deriv_sigma_buffer_->mutable_cpu_data();
	for (int i = 0; i < deriv_sigma_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }
  }
}

TYPED_TEST(GaussConvolutionLayerTest, TestCuDNNKernelPrecompute) {

	if (Caffe::mode() == Caffe::CPU)
		return;
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);

  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
	  layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(7);
  convolution_param->add_stride(1);
  convolution_param->add_pad(3);

  convolution_param->add_number_gauss(2);

  convolution_param->set_num_output(32);

  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_std(0.01);

  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  convolution_param->mutable_sigma_filler()->set_type("constant");
  convolution_param->mutable_sigma_filler()->set_value(0.8);

  convolution_param->set_gmm_component_border_bound(1.5);
  convolution_param->set_gmm_sigma_lower_bound(0.5);

  convolution_param->set_gmm_weight_normalization(false);
  convolution_param->set_gmm_gauss_normalization(true);
  for (int gmm_sqrt_norm = 0; gmm_sqrt_norm <= 1; gmm_sqrt_norm++)
  {
	convolution_param->set_gmm_square_gauss_normalization((bool)gmm_sqrt_norm);

	shared_ptr<CuDNNGaussianConvLayer<Dtype> > layer(new CuDNNGaussianConvLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

	// RUN default GPU version
	layer->precompute_guassian_weights_gpu(true); // run CPU first since it will force buffers to cpu and would cause errors if gpu would be allocated first

	// store output
	Blob<Dtype> weight_cpu, deriv_error_cpu, deriv_weight_cpu, deriv_mu1_cpu, deriv_mu2_cpu, deriv_sigma_cpu;

	weight_cpu.CopyFrom(*layer->weight_buffer_, false, true);
	deriv_error_cpu.CopyFrom(*layer->deriv_error_buffer_, false, true);
	deriv_weight_cpu.CopyFrom(*layer->deriv_weight_buffer_, false, true);
	deriv_mu1_cpu.CopyFrom(*layer->deriv_mu1_buffer_, false, true);
	deriv_mu2_cpu.CopyFrom(*layer->deriv_mu2_buffer_, false, true);
	deriv_sigma_cpu.CopyFrom(*layer->deriv_sigma_buffer_, false, true);

	// RUN optimized GPU version

	layer->get_weight_filters();
	layer->get_weight_derivative_filters(layer->kernel_buf.deriv_weight);
	layer->get_mu1_derivative_filters(layer->kernel_buf.deriv_mu1, layer->kernel_buf.deriv_weight);
	layer->get_mu2_derivative_filters(layer->kernel_buf.deriv_mu2, layer->kernel_buf.deriv_weight);
	layer->get_sigma_derivative_filters(layer->kernel_buf.deriv_sigma, layer->kernel_buf.deriv_weight);


	// Check both versions
	Dtype* data_gpu;
	Dtype* data_cpu;


	data_cpu = weight_cpu.mutable_cpu_data();
	data_gpu = layer->kernel_buf.weights->mutable_cpu_data();
	for (int i = 0; i < deriv_error_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_weight_cpu.mutable_cpu_data();
	data_gpu = layer->kernel_buf.deriv_weight->mutable_cpu_data();
	for (int i = 0; i < deriv_weight_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_mu1_cpu.mutable_cpu_data();
	data_gpu = layer->kernel_buf.deriv_mu1->mutable_cpu_data();
	for (int i = 0; i < deriv_mu1_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_mu2_cpu.mutable_cpu_data();
	data_gpu = layer->kernel_buf.deriv_mu2->mutable_cpu_data();
	for (int i = 0; i < deriv_mu2_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	data_cpu = deriv_sigma_cpu.mutable_cpu_data();
	data_gpu = layer->kernel_buf.deriv_sigma->mutable_cpu_data();
	for (int i = 0; i < deriv_sigma_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }

	/*
	data_cpu = deriv_error_cpu.mutable_cpu_data();
	data_gpu = layer->deriv_error_buffer_->mutable_cpu_data();
	for (int i = 0; i < deriv_error_cpu.count(); ++i) { EXPECT_NEAR(data_gpu[i], data_cpu[i], 1e-4); }
	*/

  }
}

TYPED_TEST(GaussConvolutionLayerTest, TestSeperableConvolution) {

 if (Caffe::mode() == Caffe::GPU)
	 return;
  typedef typename TypeParam::Dtype Dtype;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  blob_bottom_vec_.push_back(this->blob_bottom_3_);
  blob_top_vec_.push_back(this->blob_top_3_);

  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
	  layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(7);
  convolution_param->add_stride(1);
  convolution_param->add_pad(3);

  convolution_param->add_number_gauss(2);

  convolution_param->set_num_output(16);

  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_std(0.01);

  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  convolution_param->mutable_sigma_filler()->set_type("constant");
  convolution_param->mutable_sigma_filler()->set_value(0.8);

  convolution_param->set_gmm_component_border_bound(1.5);
  convolution_param->set_gmm_sigma_lower_bound(0.5);

  convolution_param->set_gmm_weight_normalization(false);
  convolution_param->set_gmm_gauss_normalization(true);


  for (int gmm_sqrt_norm = 0; gmm_sqrt_norm < 1; gmm_sqrt_norm++) {
	convolution_param->set_gmm_square_gauss_normalization((bool)gmm_sqrt_norm);

	convolution_param->set_gmm_seperable_forward_pass(false);
	shared_ptr<GaussianConvLayer<Dtype> > layer(new GaussianConvLayer<Dtype>(layer_param));

	layer->SetUp(blob_bottom_vec_, blob_top_vec_);

	// RUN non-seperable
	layer->Forward(blob_bottom_vec_, blob_top_vec_);

	// store output
	Blob<Dtype> top_org;

	top_org.CopyFrom(*blob_top_vec_[0], false, true);

	layer->use_gmm_seperable_kernels = true;

	// RUN seperable version
	layer->Forward(blob_bottom_vec_, blob_top_vec_);

	// Check both versions
	Dtype* data_non_sep = top_org.mutable_cpu_data();
	Dtype* data_sep = blob_top_vec_[0]->mutable_cpu_data();
	for (int i = 0; i < top_org.count(); ++i) { EXPECT_NEAR(data_non_sep[i], data_sep[i], 1e-4); }

	Dtype* data_diff = top_org.mutable_cpu_diff();
	caffe_sub(top_org.count(), data_non_sep, data_sep, data_diff);
	Dtype mean_error = caffe_cpu_asum(top_org.count(), data_diff) /top_org.count();
	EXPECT_NEAR(mean_error, 0, 1e-4);
  }
}

TYPED_TEST(GaussConvolutionLayerTest, TestCuDNNConvolution) {

 if (Caffe::mode() == Caffe::CPU)
	 return;

  typedef typename TypeParam::Dtype Dtype;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  vector<Blob<Dtype>*> blob_bottom_vec_org_;
  vector<Blob<Dtype>*> blob_top_vec_org_;

  vector<bool> propagate_down;

  if (sizeof(Dtype) > 4)
	  return;

  propagate_down.push_back(true);
  blob_bottom_vec_.push_back(this->blob_bottom_3_);
  blob_top_vec_.push_back(this->blob_top_3_);

  FillerParameter filler_param;
  filler_param.set_value(0.1);

  Blob<Dtype>* blob_top_diff = new Blob<Dtype>();

  Blob<Dtype>* blob_bottom_3_org_ = new Blob<Dtype>();
  Blob<Dtype>* blob_top_3_org_ = new Blob<Dtype>();

  blob_bottom_3_org_->CopyFrom(*this->blob_bottom_3_, false, true);

  blob_bottom_vec_org_.push_back(blob_bottom_3_org_);
  blob_top_vec_org_.push_back(blob_top_3_org_);


  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
	  layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->add_pad(1);

  convolution_param->add_number_gauss(2);

  convolution_param->set_num_output(8);

  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_std(0.01);

  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  convolution_param->mutable_sigma_filler()->set_type("constant");
  convolution_param->mutable_sigma_filler()->set_value(0.8);

  convolution_param->set_gmm_component_border_bound(1.5);
  convolution_param->set_gmm_sigma_lower_bound(0.5);

  convolution_param->set_gmm_weight_normalization(false);
  convolution_param->set_gmm_gauss_normalization(true);

  for (int gmm_sqrt_norm = 0; gmm_sqrt_norm < 1; gmm_sqrt_norm++) {
	  convolution_param->set_gmm_square_gauss_normalization((bool)gmm_sqrt_norm);

	shared_ptr<BaseGaussianConvLayer<Dtype> > layer(new CuDNNGaussianConvLayer<Dtype>(layer_param));
	shared_ptr<BaseGaussianConvLayer<Dtype> > layer_org(new GaussianConvLayer<Dtype>(layer_param));

	layer->SetUp(blob_bottom_vec_, blob_top_vec_);
	layer_org->SetUp(blob_bottom_vec_org_, blob_top_vec_org_);

	layer_org->param_buffer_w_->CopyFrom(*layer->param_buffer_w_, false, false);
	layer_org->param_buffer_mu1_->CopyFrom(*layer->param_buffer_mu1_, false, false);
	layer_org->param_buffer_mu2_->CopyFrom(*layer->param_buffer_mu2_, false, false);
	layer_org->param_buffer_sigma_->CopyFrom(*layer->param_buffer_sigma_, false, false);
	layer_org->param_buffer_bias_->CopyFrom(*layer->param_buffer_bias_, false, false);

	// RUN forward
	layer->Forward(blob_bottom_vec_, blob_top_vec_);

	layer_org->Forward(blob_bottom_vec_org_, blob_top_vec_org_);

	blob_top_diff->ReshapeLike(*blob_top_vec_[0]);
	GaussianFiller<Dtype> filler(filler_param);
	filler.Fill(blob_top_diff);

	caffe_copy( blob_top_vec_[0]->count(), blob_top_diff->cpu_data(), blob_top_vec_[0]->mutable_cpu_diff());

	blob_top_vec_[0]->cpu_data();
	blob_top_vec_[0]->gpu_diff();

	blob_top_vec_org_[0]->CopyFrom(*blob_top_vec_[0], true, true);

//	blob_top_vec_org_[0]->cpu_diff();

	const Dtype* gpu_data = blob_top_vec_[0]->cpu_data();
	const Dtype* cpu_data = blob_top_vec_org_[0]->cpu_data();
	for (int i = 0; i < blob_top_vec_[0]->count(); ++i) { EXPECT_NEAR(gpu_data[i], cpu_data[i], 1e-4); }

	// set all back-propagated error values to 1 to ease debugging
//	caffe_gpu_set(blob_top_vec_org_[0]->count(), (Dtype)3.0, blob_top_vec_org_[0]->mutable_gpu_diff());
//	caffe_gpu_set(blob_top_vec_[0]->count(), (Dtype)3.0, blob_top_vec_[0]->mutable_gpu_diff());


//	caffe_gpu_set(blob_bottom_vec_org_[0]->count(), (Dtype)2.0, blob_bottom_vec_org_[0]->mutable_gpu_data());
//	caffe_gpu_set(blob_bottom_vec_[0]->count(), (Dtype)2.0, blob_bottom_vec_[0]->mutable_gpu_data());

	// RUN backward

	layer->Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);

	layer_org->Backward(blob_top_vec_org_, propagate_down, blob_bottom_vec_org_);

	const Dtype* gpu_diff = blob_bottom_vec_[0]->cpu_diff();
	const Dtype* cpu_diff = blob_bottom_vec_org_[0]->cpu_diff();
	for (int i = 0; i < blob_bottom_vec_[0]->count(); ++i) { EXPECT_NEAR(gpu_diff[i], cpu_diff[i], 1e-4); }

//	layer->param_buffer_w_->cpu_data();
//	layer_org->param_buffer_w_->cpu_data();

	compare_blobs(*layer->param_buffer_w_, *layer_org->param_buffer_w_, true, (Dtype)1e-4);
	compare_blobs(*layer->param_buffer_mu1_, *layer_org->param_buffer_mu1_, true, (Dtype)1e-4);
	compare_blobs(*layer->param_buffer_mu2_, *layer_org->param_buffer_mu2_, true, (Dtype)1e-4);
	compare_blobs(*layer->param_buffer_sigma_, *layer_org->param_buffer_sigma_, true, (Dtype)1e-4);

  }

  delete blob_bottom_3_org_;
  delete blob_top_3_org_;
  delete blob_top_diff;
}
TYPED_TEST(GaussConvolutionLayerTest, TestCuDNNConvolutionExtensive) {

 if (Caffe::mode() == Caffe::CPU)
	 return;

  typedef typename TypeParam::Dtype Dtype;

  if (sizeof(Dtype) > 4)
 	  return;

  // run with different combinations of settings
  // num images: 1, 2, 3, 5, 8, 11
  // num subfeatures: 1, 2, 3, 5, 8, 11
  // width: 4, 9, 16, 32, 33,
  // height: 4, 9, 16, 32, 33,
  // num features: 4, 6, 8, 16
  // num gauss: 2,3,4

  vector<int> num_imgs_args = {1, 11};
  vector<int> num_subfeat_args = {1, 2, 3, 8, 11};
  vector<int> width_args = {4, 9, 32, 33};
  vector<int> height_args = {4, 9, 32, 33,};
  vector<int> num_feat_args = {4, 16};
  vector<int> num_gauss_args = {2,3,4}; // do not forget, this is squared, so in effect with 4,9 and 16 gauss per filter

  for (int img_i = 0; img_i < num_imgs_args.size(); ++img_i) {
  for (int subfeat_i = 0; subfeat_i < num_subfeat_args.size(); ++subfeat_i) {
  for (int width_i = 0; width_i < width_args.size(); ++width_i) {
  for (int height_i = 0; height_i < height_args.size(); ++height_i) {
  for (int feat_i = 0; feat_i < num_feat_args.size(); ++feat_i) {
  for (int gauss_i = 0; gauss_i < num_gauss_args.size(); ++gauss_i) {

  int num_imgs = num_imgs_args[img_i];
  int num_subfeat = num_subfeat_args[subfeat_i];
  int width = width_args[width_i];
  int height = height_args[height_i];
  int num_feat = num_feat_args[feat_i];
  int num_gauss = num_gauss_args[gauss_i];

  std::cout << "testing num_imgs " << num_imgs << ", num_subfeat " << num_subfeat << ", width " << width << ", height " << height << ", num_feat " << num_feat << ",num_gauss " << num_gauss << std::endl;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  vector<Blob<Dtype>*> blob_bottom_vec_org_;
  vector<Blob<Dtype>*> blob_top_vec_org_;

  vector<bool> propagate_down;

  Blob<Dtype>* blob_bottom_3_ = new Blob<Dtype>(num_imgs, num_subfeat, height, width);
  Blob<Dtype>* blob_top_3_ = new Blob<Dtype>();

  {
	  FillerParameter filler_param;
	  filler_param.set_value(1.);
	  GaussianFiller<Dtype> filler(filler_param);
	  filler.Fill(blob_bottom_3_);
  }
  propagate_down.push_back(true);
  blob_bottom_vec_.push_back(blob_bottom_3_);
  blob_top_vec_.push_back(blob_top_3_);

  FillerParameter filler_param;
  filler_param.set_value(0.1);

  Blob<Dtype>* blob_top_diff = new Blob<Dtype>();

  Blob<Dtype>* blob_bottom_3_org_ = new Blob<Dtype>();
  Blob<Dtype>* blob_top_3_org_ = new Blob<Dtype>();

  blob_bottom_3_org_->CopyFrom(*blob_bottom_3_, false, true);

  blob_bottom_vec_org_.push_back(blob_bottom_3_org_);
  blob_top_vec_org_.push_back(blob_top_3_org_);


  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
	  layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->add_pad(1);

  convolution_param->add_number_gauss(num_gauss);

  convolution_param->set_num_output(num_feat);

  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_std(0.01);

  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  convolution_param->mutable_sigma_filler()->set_type("constant");
  convolution_param->mutable_sigma_filler()->set_value(0.8);

  convolution_param->set_gmm_component_border_bound(1.5);
  convolution_param->set_gmm_sigma_lower_bound(0.5);

  convolution_param->set_gmm_weight_normalization(false);
  convolution_param->set_gmm_gauss_normalization(true);

  for (int gmm_sqrt_norm = 0; gmm_sqrt_norm < 1; gmm_sqrt_norm++) {
	  convolution_param->set_gmm_square_gauss_normalization((bool)gmm_sqrt_norm);

	shared_ptr<BaseGaussianConvLayer<Dtype> > layer(new CuDNNGaussianConvLayer<Dtype>(layer_param));
	shared_ptr<BaseGaussianConvLayer<Dtype> > layer_org(new GaussianConvLayer<Dtype>(layer_param));

	layer->SetUp(blob_bottom_vec_, blob_top_vec_);
	layer_org->SetUp(blob_bottom_vec_org_, blob_top_vec_org_);

	layer_org->param_buffer_w_->CopyFrom(*layer->param_buffer_w_, false, false);
	layer_org->param_buffer_mu1_->CopyFrom(*layer->param_buffer_mu1_, false, false);
	layer_org->param_buffer_mu2_->CopyFrom(*layer->param_buffer_mu2_, false, false);
	layer_org->param_buffer_sigma_->CopyFrom(*layer->param_buffer_sigma_, false, false);
	layer_org->param_buffer_bias_->CopyFrom(*layer->param_buffer_bias_, false, false);

	// RUN forward
	layer->Forward(blob_bottom_vec_, blob_top_vec_);

	layer_org->Forward(blob_bottom_vec_org_, blob_top_vec_org_);

	blob_top_diff->ReshapeLike(*blob_top_vec_[0]);
	GaussianFiller<Dtype> filler(filler_param);
	filler.Fill(blob_top_diff);

	caffe_copy( blob_top_vec_[0]->count(), blob_top_diff->cpu_data(), blob_top_vec_[0]->mutable_cpu_diff());

	blob_top_vec_[0]->cpu_data();
	blob_top_vec_[0]->gpu_diff();

	blob_top_vec_org_[0]->CopyFrom(*blob_top_vec_[0], true, true);

//	blob_top_vec_org_[0]->cpu_diff();

	const Dtype* gpu_data = blob_top_vec_[0]->cpu_data();
	const Dtype* cpu_data = blob_top_vec_org_[0]->cpu_data();
	for (int i = 0; i < blob_top_vec_[0]->count(); ++i) { EXPECT_NEAR(gpu_data[i], cpu_data[i], 1e-4); }

	// set all back-propagated error values to 1 to ease debugging
	//caffe_gpu_set(blob_top_vec_org_[0]->count(), (Dtype)1.0, blob_top_vec_org_[0]->mutable_gpu_diff());
	//caffe_gpu_set(blob_top_vec_[0]->count(), (Dtype)1.0, blob_top_vec_[0]->mutable_gpu_diff());

	// RUN backward

	layer->Backward(blob_top_vec_, propagate_down, blob_bottom_vec_);

	layer_org->Backward(blob_top_vec_org_, propagate_down, blob_bottom_vec_org_);

	const Dtype* gpu_diff = blob_bottom_vec_[0]->cpu_diff();
	const Dtype* cpu_diff = blob_bottom_vec_org_[0]->cpu_diff();
	for (int i = 0; i < blob_bottom_vec_[0]->count(); ++i) { EXPECT_NEAR(gpu_diff[i], cpu_diff[i], 1e-4); }

//	layer->param_buffer_w_->cpu_data();
//	layer_org->param_buffer_w_->cpu_data();

	compare_blobs(*layer->param_buffer_w_, *layer_org->param_buffer_w_, true, (Dtype)1e-4);
	compare_blobs(*layer->param_buffer_mu1_, *layer_org->param_buffer_mu1_, true, (Dtype)1e-4);
	compare_blobs(*layer->param_buffer_mu2_, *layer_org->param_buffer_mu2_, true, (Dtype)1e-4);
	compare_blobs(*layer->param_buffer_sigma_, *layer_org->param_buffer_sigma_, true, (Dtype)1e-4);

  }

  delete blob_top_3_;
  delete blob_bottom_3_;
  delete blob_bottom_3_org_;
  delete blob_top_3_org_;
  delete blob_top_diff;

  } } } } } }

}

TYPED_TEST(GaussConvolutionLayerTest, TestCuDNNComponentsMerging) {

	if (Caffe::mode() == Caffe::CPU)
		return;
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);

  const int S = this->blob_bottom_2_->channels();
  const int F = 4;
  const int G = 2;

  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
	  layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(7);
  convolution_param->add_stride(1);
  convolution_param->add_pad(3);

  convolution_param->add_number_gauss(G);
  convolution_param->add_number_gauss(1);

  convolution_param->set_num_output(F);

  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_weight_filler()->set_std(0.01);

  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  convolution_param->mutable_sigma_filler()->set_type("constant");
  convolution_param->mutable_sigma_filler()->set_value(0.8);

  convolution_param->set_gmm_component_border_bound(1.5);
  convolution_param->set_gmm_sigma_lower_bound(0.5);

  convolution_param->set_gmm_weight_normalization(false);
  convolution_param->set_gmm_gauss_normalization(true);

  convolution_param->set_gmm_square_gauss_normalization(true);

  convolution_param->set_gmm_merge_threshold(0.25);
  {



	shared_ptr<CuDNNGaussianConvLayer<Dtype> > layer(new CuDNNGaussianConvLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);


	Blob<Dtype> w_blob_org, mu1_blob_org, mu2_blob_org, sigma_blob_org;

	// initialize with predefined values
	Dtype w_fixed[24] = {0.0572,-0.0474,0.0674,0.0398,0.0621,-0.0372,0.0649,0.0456,0.0251,0.0471,0.0526,-0.0762,0.0349,0.0180,0.0879,-0.0651,0.0336,-0.0493,0.0268,-0.0543,0.0195,-0.0593,0.0258,-0.0788};
	Dtype mu1_fixed[24] = {2.9007,2.6520,2.8522,2.4873,3.3834,3.2959,3.5147,3.4099,2.5391,2.9612,2.4954,2.2543,3.4901,3.9438,3.5789,3.5099,2.3823,2.0228,2.3286,2.2390,3.7147,3.6248,3.7756,3.8493};
	Dtype mu2_fixed[24] = {3.3636,3.4184,3.3023,2.2308,3.3255,3.3340,3.2989,2.1993,3.1086,3.1973,2.9114,2.9175,3.2106,3.0928,2.7739,3.0882,3.2877,3.0428,3.2214,2.8396,3.1918,3.0179,3.3646,2.5582};
	Dtype sigma_fixed[24] = {0.6166,0.6610,0.5975,0.7559,0.6365,0.7319,0.6297,0.7358,0.6671,0.5842,0.7480,0.7521,0.6317,0.7678,0.6967,0.7771,0.7639,0.8072,0.6916,0.7846,0.8464,0.8034,0.7022,0.7525};

	Dtype* w_set = layer->param_buffer_w_->mutable_cpu_data();
	Dtype* mu1_set = layer->param_buffer_mu1_->mutable_cpu_data();
	Dtype* mu2_set = layer->param_buffer_mu2_->mutable_cpu_data();
	Dtype* sigma_set = layer->param_buffer_sigma_->mutable_cpu_data();

	memcpy(w_set, w_fixed, sizeof(Dtype) * 24);
	memcpy(mu1_set, mu1_fixed, sizeof(Dtype) * 24);
	memcpy(mu2_set, mu2_fixed, sizeof(Dtype) * 24);
	memcpy(sigma_set, sigma_fixed, sizeof(Dtype) * 24);

	w_blob_org.CopyFrom(*layer->param_buffer_w_, false, true);
	mu1_blob_org.CopyFrom(*layer->param_buffer_mu1_, false, true);
	mu2_blob_org.CopyFrom(*layer->param_buffer_mu2_, false, true);
	sigma_blob_org.CopyFrom(*layer->param_buffer_sigma_, false, true);

	Blob<Dtype> scores_blob(3,S,G,F);

	// run components merging
	layer->merge_components(&scores_blob);

	const Dtype* scores = scores_blob.cpu_data();
	const Dtype* score_dist = scores_blob.cpu_data() + scores_blob.offset(0);
	const Dtype* score_hell = scores_blob.cpu_data() + scores_blob.offset(1);
	const Dtype* score_KL = scores_blob.cpu_data() + scores_blob.offset(2);


	Dtype gt[24*3] = {0.234451, 0.421731, 0.438918, 0.852183, 0.234451, 0.421731, 0.438918, 0.852183, 0.914805, 0.976423, 1.192879, 1.605670, 0.914805, 0.976423, 1.192879, 1.605670, 1.784487, 2.567024, 2.114315, 2.672252, 1.784487, 2.567024, 2.114315, 2.672252, 0.072152, 0.105056, 0.136109, 0.174391, 0.072152, 0.105056, 0.136109, 0.174391, 0.237913, 0.244742, 0.249239, 0.290713, 0.237913, 0.244742, 0.249239, 0.290713, 0.292355, 0.390305, 0.419695, 0.432039, 0.292355, 0.390305, 0.419695, 0.432039, 0.290339, 0.403353, 0.556120, 0.787755, 0.290339, 0.403353, 0.556120, 0.787755, 1.149327, 0.890907, 1.234077, 1.330498, 1.149327, 0.890907, 1.234077, 1.330498, 1.255300, 1.988571, 2.144196, 2.361373, 1.255300, 1.988571, 2.144196, 2.361373 };

	for (int i = 0; i < scores_blob.count(); ++i)
		EXPECT_NEAR(scores[i], gt[i], 1e-2);
/*
	const Dtype* w = layer->param_buffer_w_->cpu_data();
	const Dtype* mu1 = layer->param_buffer_mu1_->cpu_data();
	const Dtype* mu2 = layer->param_buffer_mu2_->cpu_data();
	const Dtype* sigma = layer->param_buffer_sigma_->cpu_data();

	const Dtype* w_org = w_blob_org.cpu_data();
	const Dtype* mu1_org = mu1_blob_org.cpu_data();
	const Dtype* mu2_org = mu2_blob_org.cpu_data();
	const Dtype* sigma_org = sigma_blob_org.cpu_data();

	printf("Weights (org vs fixed -- dist, hellinger, KL-dist): \n");
	for (int i = 0; i < w_blob_org.count(); ++i)
		printf("%f, %f -- %f, %f, %f\n",w_org[i],w[i], score_dist[i],score_hell[i],score_KL[i]);

	printf("Mean 1 (org vs fixed ): \n");
	for (int i = 0; i < w_blob_org.count(); ++i)
		printf("%f, %f\n",mu1_org[i],mu1[i] );

	printf("Mean 2 (org vs fixed ): \n");
	for (int i = 0; i < w_blob_org.count(); ++i)
		printf("%f, %f\n",mu2_org[i],mu2[i] );

	printf("Sigma (org vs fixed ): \n");
	for (int i = 0; i < w_blob_org.count(); ++i)
		printf("%f, %f\n",sigma_org[i],sigma[i] );
*/
  }
}

#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

TYPED_TEST(GaussConvolutionLayerTest, TestFastGaussForward) {

    Caffe::SetDevice(0);

    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;

    // evaluate size settings
    const int N = 128;
    const int F = 32;
    const int S = 32;
    const int G = 2;
    const int W = 64;
    const int H = 32;

    const bool use_interpolation = true;

    const int kernel_size = 17;

    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_output(N,F,H,W);
    Blob<float> blob_output_cpu(N,F,H,W);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_std(1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);

    const_zero_float_filer.Fill(&blob_output);

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(1);
    offset_filler_param.set_max(kernel_size-2);
    UniformFiller<float> offset_filler(offset_filler_param);

    LayerParameter layer_param;

    ConvolutionParameter* gauss_convolution_param = layer_param.mutable_convolution_param();

    gauss_convolution_param->add_kernel_size(kernel_size);
    gauss_convolution_param->add_stride(1);
    gauss_convolution_param->add_pad(kernel_size/2);

    gauss_convolution_param->add_number_gauss(G);
    gauss_convolution_param->add_number_gauss(1);

    gauss_convolution_param->set_num_output(F);

    gauss_convolution_param->mutable_weight_filler()->set_type("gaussian");
    gauss_convolution_param->mutable_weight_filler()->set_std(0.1);

    gauss_convolution_param->mutable_bias_filler()->set_type("constant");
    gauss_convolution_param->mutable_bias_filler()->set_value(0);

    gauss_convolution_param->mutable_mu_filler()->set_type("constant");
    gauss_convolution_param->mutable_mu_filler()->set_value(0);

    gauss_convolution_param->mutable_sigma_filler()->set_type("constant");
    gauss_convolution_param->mutable_sigma_filler()->set_value(0.8);

    gauss_convolution_param->set_gmm_component_border_bound(100);
    gauss_convolution_param->set_gmm_sigma_lower_bound(0.5);

    gauss_convolution_param->set_gmm_weight_normalization(false);
    gauss_convolution_param->set_gmm_gauss_normalization(true);
    gauss_convolution_param->set_gmm_square_gauss_normalization(false);

    FastAproxGaussianConvLayer<float> layer(layer_param);

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_cpu;

    blob_bottom_vec.push_back(&blob_input);

    blob_top_vec.push_back(&blob_output);
    blob_top_vec_cpu.push_back(&blob_output_cpu);

    layer.SetUp(blob_bottom_vec, blob_top_vec);

    // override offset data with random values - must be done after SetUp
    offset_filler.Fill(layer.param_buffer_mu1_.get());
    offset_filler.Fill(layer.param_buffer_mu2_.get());

    // perform forward pass with CPU version
    layer.Forward_cpu(blob_bottom_vec, blob_top_vec_cpu);

    // perform forward pass with GPU version
    layer.Forward_gpu(blob_bottom_vec, blob_top_vec);

    const float* output_cpu = blob_top_vec_cpu[0]->cpu_data();
    const float* output_gpu = blob_top_vec[0]->cpu_data();

    // verify data with CPU version
    int found_invalid = 0;

    double diff = 0;
    double max_diff = 0;

    for (int n = 0; n < N; ++n){
        for (int f = 0; f < F; ++f) {
            for (int i = 0; i < H * W; ++i) {
                int index = (n * F + f )* H * W + i;
                float val = output_gpu[index];
                float GT_VALUE = output_cpu[index];

                // interpolation at the right edge excludes one pixel so ignore those pixels
                if (std::abs(val - GT_VALUE) / std::abs(GT_VALUE) > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && i % W < (W -1)) {
                    if (found_invalid < 10)
                        printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index,  n, f, i / W, i % W, GT_VALUE);
                    found_invalid++;

                    double current_diff = std::abs(val - GT_VALUE) / std::abs(GT_VALUE);

                    diff += current_diff;

                    max_diff = std::max(max_diff, current_diff);
                }
            }
        }
    }
    if (found_invalid > 0) {
        diff /= found_invalid;
        printf("found num of invalid output vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid, blob_output.count(), diff, max_diff);
    }

    // report fail only if enough samples with big enough diff
    EXPECT_NEAR(diff, 0, 1e-3);
    EXPECT_NEAR(found_invalid/(float)blob_output.count(), 0, 1e-2);
}

TYPED_TEST(GaussConvolutionLayerTest, TestFastGaussForwardWithGroundtruth) {

    Caffe::SetDevice(0);

    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;

    // evaluate size settings
    const int N = 128;
    const int F = 32;
    const int S = 16;
    const int G = 2;
    const int W = 32;
    const int H = 32;

    const bool use_interpolation = true;

    const int kernel_size = 9;

    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_output(N,F,H,W);
    Blob<float> blob_output_gt(N,F,H,W);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_std(1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);

    const_zero_float_filer.Fill(&blob_output);
    const_zero_float_filer.Fill(&blob_output_gt);

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(3);
    offset_filler_param.set_max(kernel_size-3);
    UniformFiller<float> offset_filler(offset_filler_param);

    LayerParameter layer_param;

    ConvolutionParameter* gauss_convolution_param = layer_param.mutable_convolution_param();

    gauss_convolution_param->add_kernel_size(kernel_size);
    gauss_convolution_param->add_stride(1);
    gauss_convolution_param->add_pad(kernel_size/2);

    gauss_convolution_param->add_number_gauss(G);
    gauss_convolution_param->add_number_gauss(1);

    gauss_convolution_param->set_num_output(F);

    gauss_convolution_param->mutable_weight_filler()->set_type("gaussian");
    gauss_convolution_param->mutable_weight_filler()->set_std(0.1);

    gauss_convolution_param->mutable_bias_filler()->set_type("constant");
    gauss_convolution_param->mutable_bias_filler()->set_value(0);

    gauss_convolution_param->mutable_mu_filler()->set_type("constant");
    gauss_convolution_param->mutable_mu_filler()->set_value(0);

    // NOTE: we use small sigma to effecitvly use identity for gaussian matrix; fast version will not be able to handle
    // border values well since gaussian is not computed outside of valid pixesl (TODO: we could chnaged that by doing gaussian pre-filtering on more padding)
    gauss_convolution_param->mutable_sigma_filler()->set_type("constant");
    gauss_convolution_param->mutable_sigma_filler()->set_value(0.1);

    gauss_convolution_param->set_gmm_component_border_bound(0);
    gauss_convolution_param->set_gmm_sigma_lower_bound(0.1);

    gauss_convolution_param->set_gmm_weight_normalization(false);
    gauss_convolution_param->set_gmm_gauss_normalization(true);
    gauss_convolution_param->set_gmm_square_gauss_normalization(false);

    FastAproxGaussianConvLayer<float> layer(layer_param);
    CuDNNGaussianConvLayer<float> layer_gt(layer_param);

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_gt;

    blob_bottom_vec.push_back(&blob_input);

    blob_top_vec.push_back(&blob_output);
    blob_top_vec_gt.push_back(&blob_output_gt);

    layer.SetUp(blob_bottom_vec, blob_top_vec);

    layer_gt.SetUp(blob_bottom_vec, blob_top_vec_gt);

    const_zero_float_filer.Fill(&blob_output);
    const_zero_float_filer.Fill(&blob_output_gt);
    const_one_filer.Fill(&blob_input);

    // override offset data with random values - must be done after SetUp
    offset_filler.Fill(layer.param_buffer_mu1_.get());
    offset_filler.Fill(layer.param_buffer_mu2_.get());

    // discretize mu1,mu2 for comparision with groundtruth version
    // copy w,mu1,mu2 and sigma settings to layer_gt

    float *w_data = layer.param_buffer_w_->mutable_cpu_data();
    float *mu1_data = layer.param_buffer_mu1_->mutable_cpu_data();
    float *mu2_data = layer.param_buffer_mu2_->mutable_cpu_data();
    float *sigma_data = layer.param_buffer_sigma_->mutable_cpu_data();

    float *w_data_gt = layer_gt.param_buffer_w_->mutable_cpu_data();
    float *mu1_data_gt = layer_gt.param_buffer_mu1_->mutable_cpu_data();
    float *mu2_data_gt = layer_gt.param_buffer_mu2_->mutable_cpu_data();
    float *sigma_data_gt = layer_gt.param_buffer_sigma_->mutable_cpu_data();

    for (int s = 0; s < S; s++) {
        for (int g = 0; g < G; g++) {
            for (int f = 0; f < F; f++) {
                // discretize the offsets
                mu1_data[OFFSET(0, s, g, f, 1, S, G, F)] = floor(mu1_data[OFFSET(0, s, g, f, 1, S, G, F)]);
                mu2_data[OFFSET(0, s, g, f, 1, S, G, F)] = floor(mu2_data[OFFSET(0, s, g, f, 1, S, G, F)]);

                // and set the same values for groundtruth layer
                w_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = w_data[OFFSET(0,s,g,f, 1, S,G,F)];

                mu1_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = mu1_data[OFFSET(0, s, g, f, 1, S, G, F)];
                mu2_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = mu2_data[OFFSET(0, s, g, f, 1, S, G, F)];

                // set fixed sigma for all components in groundtruth layer
                sigma_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = sigma_data[OFFSET(0, s, g, f, 1, S, G, F)];
            }
        }
    }


    // perform forward pass with CPU version
    layer_gt.Forward_gpu(blob_bottom_vec, blob_top_vec_gt);

    // perform forward pass with GPU version
    layer.Forward_gpu(blob_bottom_vec, blob_top_vec);

    float* output_gt = blob_top_vec_gt[0]->mutable_cpu_data();
    float* output_gpu = blob_top_vec[0]->mutable_cpu_data();

    // verify data with groundtruth version
    int found_invalid = 0;

    double diff = 0;
    double max_diff = 0;

    for (int n = 0; n < N; ++n){
        for (int f = 0; f < F; ++f) {
            for (int i = 0; i < H * W; ++i) {
                int index = (n * F + f )* H * W + i;
                float val = output_gpu[index];
                float GT_VALUE = output_gt[index];

                // interpolation at the right edge excludes one pixel so ignore those pixels
                if (std::abs(val - GT_VALUE) / std::abs(GT_VALUE) > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && i % W < (W -1)) {
                    if (found_invalid < 10)
                        printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index,  n, f, i / W, i % W, GT_VALUE);
                    found_invalid++;

                    double current_diff = std::abs(val - GT_VALUE) / std::abs(GT_VALUE);

                    diff += current_diff;

                    max_diff = std::max(max_diff, current_diff);
                }
            }
        }
    }
    if (found_invalid > 0) {
        diff /= found_invalid;
        printf("found num of invalid output vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid, blob_output.count(), diff, max_diff);
    }

    // report fail only if enough samples with big enough diff
    EXPECT_NEAR(diff, 0, 1e-3);
    EXPECT_NEAR(found_invalid/(float)blob_output.count(), 0, 1e-2);
}

TYPED_TEST(GaussConvolutionLayerTest, TestFastGaussBackward) {


    typedef typename TypeParam::Dtype Dtype;

    Caffe::SetDevice(0);

    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;
    // evaluate size settings

    const int N = 128;
    const int F = 32;
    const int S = 32;
    const int G = 2;
    const int W = 64;
    const int H = 32;

    // number of Guassian learning parameters we have (w,mu1,mu2,sigma)
    // for each parameter we need convolution of input data with specific kernel
    const int K = 3;
    const bool use_interpolation = true;
    const bool ignore_edge_gradients = true; // for cpu/gpu compatability

    const int kernel_size = 9;

    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_error(N,F,H,W);
    Blob<float> blob_output(K, S, G, F);
    Blob<float> blob_output_cpu(K, S, G, F);

    Blob<float> blob_output_error_cpu(N,S,H,W);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_value(1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);
    input_filler.Fill(&blob_error);
    caffe_rng_gaussian<float>(blob_error.count(), float(0), float(0.1), blob_error.mutable_cpu_diff());

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(4);
    offset_filler_param.set_max(kernel_size - 4);

    UniformFiller<float> offset_filler(offset_filler_param);

    const_zero_float_filer.Fill(&blob_output);

    LayerParameter layer_param;
    ConvolutionParameter* gauss_convolution_param = layer_param.mutable_convolution_param();

    gauss_convolution_param->add_kernel_size(kernel_size);
    gauss_convolution_param->add_stride(1);
    gauss_convolution_param->add_pad(kernel_size/2);

    gauss_convolution_param->add_number_gauss(G);
    gauss_convolution_param->add_number_gauss(1);

    gauss_convolution_param->set_num_output(F);

    gauss_convolution_param->mutable_weight_filler()->set_type("gaussian");
    gauss_convolution_param->mutable_weight_filler()->set_std(0.1);

    gauss_convolution_param->mutable_bias_filler()->set_type("constant");
    gauss_convolution_param->mutable_bias_filler()->set_value(0);

    gauss_convolution_param->mutable_mu_filler()->set_type("constant");
    gauss_convolution_param->mutable_mu_filler()->set_value(0);

    gauss_convolution_param->mutable_sigma_filler()->set_type("constant");
    gauss_convolution_param->mutable_sigma_filler()->set_value(0.8);

    gauss_convolution_param->set_gmm_component_border_bound(0);
    gauss_convolution_param->set_gmm_sigma_lower_bound(0.5);

    gauss_convolution_param->set_gmm_weight_normalization(false);
    gauss_convolution_param->set_gmm_gauss_normalization(true);
    gauss_convolution_param->set_gmm_square_gauss_normalization(false);

    FastAproxGaussianConvLayer<float> layer(layer_param);

    layer.ignore_edge_gradients_ = ignore_edge_gradients;

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_gt;
    std::vector<bool > propagate_down;
    propagate_down.push_back(true);

    blob_bottom_vec.push_back(&blob_input);
    blob_top_vec.push_back(&blob_error);

    layer.SetUp(blob_bottom_vec, blob_top_vec);

    // override offset data with random values within valid range
    offset_filler.Fill(layer.param_buffer_mu1_.get());
    offset_filler.Fill(layer.param_buffer_mu2_.get());

    caffe_gpu_set(layer.param_buffer_w_->count(), (Dtype)0, (Dtype*)layer.param_buffer_w_->mutable_gpu_diff());
    caffe_gpu_set(layer.param_buffer_mu1_->count(), (Dtype)0, (Dtype*)layer.param_buffer_mu1_->mutable_gpu_diff());
    caffe_gpu_set(layer.param_buffer_mu2_->count(), (Dtype)0, (Dtype*)layer.param_buffer_mu2_->mutable_gpu_diff());
    caffe_gpu_set(layer.param_buffer_sigma_->count(), (Dtype)0, (Dtype*)layer.param_buffer_sigma_->mutable_gpu_diff());

    layer.Backward_cpu(blob_top_vec, propagate_down, blob_bottom_vec);

    // save backproped error values
    float *backprop_error_cpu = blob_output_error_cpu.mutable_cpu_data();
    {
        caffe_copy(blob_output_error_cpu.count(), blob_bottom_vec[0]->cpu_diff(), backprop_error_cpu);
    }

    // save accumulated gradient values
    float *gradients_cpu = blob_output_cpu.mutable_cpu_data();
    {
        int num_params = layer.param_buffer_w_->count();
        if (K > 0) caffe_copy(num_params, layer.param_buffer_w_->cpu_diff(), gradients_cpu + 0 * num_params );
        if (K > 1) caffe_copy(num_params, layer.param_buffer_mu1_->cpu_diff(), gradients_cpu + 1 * num_params );
        if (K > 2) caffe_copy(num_params, layer.param_buffer_mu2_->cpu_diff(), gradients_cpu + 2 * num_params );
        if (K > 3) caffe_copy(num_params, layer.param_buffer_sigma_->cpu_diff(), gradients_cpu + 3 * num_params);
    }

    // reset values for GPU run
    caffe_gpu_set(layer.param_buffer_w_->count(), (Dtype)0, (Dtype*)layer.param_buffer_w_->mutable_gpu_diff());
    caffe_gpu_set(layer.param_buffer_mu1_->count(), (Dtype)0, (Dtype*)layer.param_buffer_mu1_->mutable_gpu_diff());
    caffe_gpu_set(layer.param_buffer_mu2_->count(), (Dtype)0, (Dtype*)layer.param_buffer_mu2_->mutable_gpu_diff());
    caffe_gpu_set(layer.param_buffer_sigma_->count(), (Dtype)0, (Dtype*)layer.param_buffer_sigma_->mutable_gpu_diff());


    layer.Backward_gpu(blob_top_vec, propagate_down, blob_bottom_vec);

    // get ptr to backproped error on gpu
    const float *backprop_error_gpu = blob_bottom_vec[0]->cpu_diff();

    // save accumulated gradient values
    float *gradients_gpu_g = blob_output.mutable_gpu_data();

    int num_params = layer.param_buffer_w_->count();
    if (K > 0) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_w_->gpu_diff(), gradients_gpu_g + 0 * num_params );
    if (K > 1) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_mu1_->gpu_diff(), gradients_gpu_g + 1 * num_params );
    if (K > 2) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_mu2_->gpu_diff(), gradients_gpu_g + 2 * num_params );
    if (K > 3) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_sigma_->gpu_diff(), gradients_gpu_g + 3 * num_params);

    // get gradients from GPU, but read them on cpu
    float *gradients_gpu = blob_output.mutable_cpu_data();

    {
        // verify accumulated gradients
        int found_invalid_backprop = 0;

        double diff_gradient = 0;
        double max_diff = 0;
        for (int k = 0; k < K; ++k) {
            for (int s = 0; s < S; ++s) {
                for (int g = 0; g < G; ++g) {
                    for (int f = 0; f < F; ++f) {
                        int idx = OFFSET(k,s,g,f, K, S,  G, F);
                        float val = gradients_gpu[idx];
                        float GT_VALUE = gradients_cpu[idx];

                        if (std::abs(val - GT_VALUE) / std::abs(GT_VALUE) > 1e-4 && std::abs(val - GT_VALUE) > 1e-7) {
                            if (found_invalid_backprop < 10)
                                printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, idx, k, s,g,f, GT_VALUE);
                            found_invalid_backprop++;
                            double current_diff = std::abs(val - GT_VALUE) / std::abs(GT_VALUE);

                            diff_gradient += current_diff;

                            max_diff = std::max(max_diff, current_diff);

                        }
                    }
                }
            }
        }

        if (found_invalid_backprop > 0) {
            diff_gradient /= found_invalid_backprop;
            printf("found num of invalid accumulated gradient vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid_backprop, K * S * G * F, diff_gradient, max_diff);
        }
        // report fail only if enough samples with big enough diff
        EXPECT_NEAR(diff_gradient, 0, 1e-2);
        EXPECT_NEAR(found_invalid_backprop/(float)(K * S * G * F), 0, 1e-2);
    }
    {
        // verify accumulated gradients
        int found_invalid_backprop = 0;

        double diff_backprop = 0;
        double max_diff = 0;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < S; ++s) {
                for (int i = 0; i < H * W; ++i) {
                    int index = (n * S + s )* H * W + i;
                    float val = backprop_error_gpu[index];
                    float GT_VALUE = backprop_error_cpu[index];

                    if (std::abs(val - GT_VALUE) / GT_VALUE > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && i % W < (W -1)) {
                        if (found_invalid_backprop < 10)
                            printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index, n, s, i / W, i % W, GT_VALUE);
                        found_invalid_backprop++;
                        diff_backprop += std::abs(val - GT_VALUE) / GT_VALUE;

                        double current_diff = std::abs(val - GT_VALUE) / GT_VALUE;

                        diff_backprop += current_diff;

                        max_diff = std::max(max_diff, current_diff);


                    }
                }
            }
        }

        if (found_invalid_backprop > 0) {
            diff_backprop /= found_invalid_backprop;
            printf("found num of invalid backproped-error vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid_backprop, N * S * H * W, diff_backprop, max_diff);
        }
        // report fail only if enough samples with big enough diff
        EXPECT_NEAR(diff_backprop, 0, 1e-3);
        EXPECT_NEAR(found_invalid_backprop/(float)(N * S * H * W), 0, 1e-2);
    }
}


TYPED_TEST(GaussConvolutionLayerTest, TestFastGaussBackwardWithGroundtruth) {


    typedef typename TypeParam::Dtype Dtype;

    Caffe::SetDevice(0);

    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;
    // evaluate size settings

    const int N = 128;
    const int F = 32;
    const int S = 32;
    const int G = 2;
    const int W = 64;
    const int H = 32;

    // number of Guassian learning parameters we have (w,mu1,mu2,sigma)
    // for each parameter we need convolution of input data with specific kernel
    const int K = 3;
    const bool use_interpolation = true;
    const bool ignore_edge_gradients = true; // for cpu/gpu compatability

    const int kernel_size = 9;

    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_error(N,F,H,W);
    Blob<float> blob_error_gt(N,F,H,W);
    Blob<float> blob_output(K, S, G, F);
    Blob<float> blob_output_gt(K, S, G, F);
    Blob<float> blob_output_error_gt(N,S,H,W);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_value(1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);

    caffe_rng_gaussian<float>(blob_error.count(), float(0), float(0.1), blob_error.mutable_cpu_diff());

    // copy error values for GT version but without borders for compatability with GPU
    float* error = blob_error.mutable_cpu_diff();
    float* error_gt = blob_error_gt.mutable_cpu_diff();
    for (int n = 0; n < N*F; ++n){
        for (int i = 0; i < H*W; ++i) {
            // set error values at the bottom/right borders to zero for comparability with FastGPU version
            error_gt[n*H*W + i] = i % W < W -1 && i / W < H-1 ? error[n*H*W + i] : 0;
        }
    }

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(3);
    offset_filler_param.set_max(kernel_size - 3);

    UniformFiller<float> offset_filler(offset_filler_param);

    const_zero_float_filer.Fill(&blob_output);

    LayerParameter layer_param;
    ConvolutionParameter* gauss_convolution_param = layer_param.mutable_convolution_param();

    gauss_convolution_param->add_kernel_size(kernel_size);
    gauss_convolution_param->add_stride(1);
    gauss_convolution_param->add_pad(kernel_size/2);

    gauss_convolution_param->add_number_gauss(G);
    gauss_convolution_param->add_number_gauss(1);

    gauss_convolution_param->set_num_output(F);

    gauss_convolution_param->mutable_weight_filler()->set_type("gaussian");
    gauss_convolution_param->mutable_weight_filler()->set_std(0.1);

    gauss_convolution_param->mutable_bias_filler()->set_type("constant");
    gauss_convolution_param->mutable_bias_filler()->set_value(0);

    gauss_convolution_param->mutable_mu_filler()->set_type("constant");
    gauss_convolution_param->mutable_mu_filler()->set_value(0);

    gauss_convolution_param->mutable_sigma_filler()->set_type("constant");
    gauss_convolution_param->mutable_sigma_filler()->set_value(0.8);

    gauss_convolution_param->set_gmm_component_border_bound(0);
    gauss_convolution_param->set_gmm_sigma_lower_bound(0.1);

    gauss_convolution_param->set_gmm_weight_normalization(false);
    gauss_convolution_param->set_gmm_gauss_normalization(true);
    gauss_convolution_param->set_gmm_square_gauss_normalization(false);

    FastAproxGaussianConvLayer<float> layer(layer_param);
    CuDNNGaussianConvLayer<float> layer_gt(layer_param);

    layer.ignore_edge_gradients_ = ignore_edge_gradients;

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_gt;
    std::vector<bool > propagate_down;
    propagate_down.push_back(true);

    blob_bottom_vec.push_back(&blob_input);
    blob_top_vec.push_back(&blob_error);
    blob_top_vec_gt.push_back(&blob_error_gt);


    layer.SetUp(blob_bottom_vec, blob_top_vec);
    layer_gt.SetUp(blob_bottom_vec, blob_top_vec);

    // override offset data with random values with a valid range
    offset_filler.Fill(layer.param_buffer_mu1_.get());
    offset_filler.Fill(layer.param_buffer_mu2_.get());

    float *w_data = layer.param_buffer_w_->mutable_cpu_data();
    float *mu1_data = layer.param_buffer_mu1_->mutable_cpu_data();
    float *mu2_data = layer.param_buffer_mu2_->mutable_cpu_data();
    float *sigma_data = layer.param_buffer_sigma_->mutable_cpu_data();

    float *w_data_gt = layer_gt.param_buffer_w_->mutable_cpu_data();
    float *mu1_data_gt = layer_gt.param_buffer_mu1_->mutable_cpu_data();
    float *mu2_data_gt = layer_gt.param_buffer_mu2_->mutable_cpu_data();
    float *sigma_data_gt = layer_gt.param_buffer_sigma_->mutable_cpu_data();

    for (int s = 0; s < S; s++) {
        for (int g = 0; g < G; g++) {
            for (int f = 0; f < F; f++) {
                // we can use only center offset otherwise we get difference in border values and gradients become different
                mu1_data[OFFSET(0, s, g, f, 1, S, G, F)] = kernel_size/2;//floor(mu1_data[OFFSET(0, s, g, f, 1, S, G, F)]);
                mu2_data[OFFSET(0, s, g, f, 1, S, G, F)] = kernel_size/2;//floor(mu2_data[OFFSET(0, s, g, f, 1, S, G, F)]);

                w_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = w_data[OFFSET(0,s,g,f, 1, S,G,F)];
                // and set the same values for groundtruth layer
                mu1_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = mu1_data[OFFSET(0, s, g, f, 1, S, G, F)];
                mu2_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = mu2_data[OFFSET(0, s, g, f, 1, S, G, F)];
                // set fixed sigma for all components in groundtruth layer
                sigma_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = sigma_data[OFFSET(0, s, g, f, 1, S, G, F)];
            }
        }
    }

    layer_gt.Backward_gpu(blob_top_vec_gt, propagate_down, blob_bottom_vec);

    // save accumulated gradient values
    float *gradients_gt = blob_output_gt.mutable_cpu_data();
    {
        int num_params = layer_gt.param_buffer_w_->count();
        if (K > 0) caffe_copy(num_params, layer_gt.param_buffer_w_->cpu_diff(), gradients_gt + 0 * num_params );
        if (K > 1) caffe_copy(num_params, layer_gt.param_buffer_mu1_->cpu_diff(), gradients_gt + 1 * num_params );
        if (K > 2) caffe_copy(num_params, layer_gt.param_buffer_mu2_->cpu_diff(), gradients_gt + 2 * num_params );
        if (K > 3) caffe_copy(num_params, layer_gt.param_buffer_sigma_->cpu_diff(), gradients_gt + 3 * num_params);
    }

    // re-run but with original top values
    layer_gt.Backward_gpu(blob_top_vec, propagate_down, blob_bottom_vec);

    // save backproped error values
    float *backprop_error_gt = blob_output_error_gt.mutable_cpu_data();
    {
        caffe_copy(blob_output_error_gt.count(), blob_bottom_vec[0]->cpu_diff(), backprop_error_gt);
    }


    layer.Backward_gpu(blob_top_vec, propagate_down, blob_bottom_vec);

    // get ptr to backproped error on gpu
    const float *backprop_error_gpu = blob_bottom_vec[0]->cpu_diff();

    // save accumulated gradient values
    float *gradients_gpu_g = blob_output.mutable_gpu_data();

    int num_params = layer.param_buffer_w_->count();
    if (K > 0) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_w_->gpu_diff(), gradients_gpu_g + 0 * num_params );
    if (K > 1) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_mu1_->gpu_diff(), gradients_gpu_g + 1 * num_params );
    if (K > 2) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_mu2_->gpu_diff(), gradients_gpu_g + 2 * num_params );
    if (K > 3) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_sigma_->gpu_diff(), gradients_gpu_g + 3 * num_params);

    // get gradients from GPU, but read them on cpu
    float *gradients_gpu = blob_output.mutable_cpu_data();

    {
        // verify accumulated gradients
        int found_invalid_gradient = 0;

        double diff_gradient = 0;
        double max_diff = 0;
        for (int k = 0; k < K; ++k) {
            for (int s = 0; s < S; ++s) {
                for (int g = 0; g < G; ++g) {
                    for (int f = 0; f < F; ++f) {
                        int idx = OFFSET(k,s,g,f, K, S,  G, F);
                        float val = gradients_gpu[idx];
                        float GT_VALUE = gradients_gt[idx];

                        if (std::abs(val - GT_VALUE) / std::abs(GT_VALUE) > 1e-3 && std::abs(val - GT_VALUE) > 1e-7) {
                            if (found_invalid_gradient < 10)
                                printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, idx, k, s,g,f, GT_VALUE);
                            found_invalid_gradient++;
                            double current_diff = std::abs(val - GT_VALUE) / std::abs(GT_VALUE);

                            diff_gradient += current_diff;

                            max_diff = std::max(max_diff, current_diff);

                        }
                    }
                }
            }
        }

        if (found_invalid_gradient > 0) {
            diff_gradient /= found_invalid_gradient;
            printf("found num of invalid accumulated gradient vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid_gradient, K*S*G*F, diff_gradient, max_diff);
        }

        // report fail only if enough samples with big enough diff
        EXPECT_NEAR(diff_gradient, 0, 1e-2);
        EXPECT_NEAR(found_invalid_gradient/(float)(N * S * H * W), 0, 1e-2);

    }
    {
        // verify accumulated gradients
        int found_invalid_backprop = 0;

        double diff_backprop = 0;
        double max_diff = 0;
        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < S; ++s) {
                for (int i = 0; i < H * W; ++i) {
                    int index = (n * S + s )* H * W + i;
                    float val = backprop_error_gpu[index];
                    float GT_VALUE = backprop_error_gt[index];

                    if (std::abs(val - GT_VALUE) / std::abs(GT_VALUE) > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && i % W < (W -1)) {
                        if (found_invalid_backprop < 10)
                            printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index, n, s, i / W, i % W, GT_VALUE);
                        found_invalid_backprop++;
                        double current_diff = std::abs(val - GT_VALUE) / std::abs(GT_VALUE);

                        diff_backprop += current_diff;

                        max_diff = std::max(max_diff, current_diff);


                    }
                }
            }
        }


        if (found_invalid_backprop > 0) {
            diff_backprop /= found_invalid_backprop;
            printf("found num of invalid backproped-error vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid_backprop, N*S*H*W, diff_backprop, max_diff);
        }
        // report fail only if enough samples with big enough diff
        EXPECT_NEAR(diff_backprop, 0, 1e-2);
        EXPECT_NEAR(found_invalid_backprop/(float)(N * S * H * W), 0, 1e-2);
    }
}


TYPED_TEST(GaussConvolutionLayerTest, DebugFastGaussConvolution) {

    Caffe::SetDevice(0);

    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;

    // evaluate size settings
    const int N = 128;
    const int F = 32;
    const int S = 3;
    const int G = 2;
    const int W = 48;
    const int H = 16;

    const bool use_interpolation = true;

    const int kernel_size = 9;

    Blob<float> blob_input(N,S,H,W);
    Blob<int> blob_offsets_x(1, S, G, F);
    Blob<int> blob_offsets_y(1, S, G, F);
    Blob<float> blob_offsets_float_x(1, S, G, F);
    Blob<float> blob_offsets_float_y(1, S, G, F);
    Blob<float> blob_weights(1, S, G, F);
    Blob<float> blob_output(N,F,H,W);
    Blob<float> blob_output_cpu(N,F,H,W);
    Blob<float> blob_output_cudnn(N,F,H,W);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_std(0.1);
    GaussianFiller<float> input_filler(rand_filler_param);

    //input_filler.Fill(&blob_input);
    //input_filler.Fill(&blob_weights);

    const_one_filer.Fill(&blob_input);
    const_one_filer.Fill(&blob_weights);


    input_filler.Fill(&blob_input);

    float* data = blob_input.mutable_cpu_data();
    for (int n = 0; n < N; ++n){
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < H * W; ++i) {
                //data[(n * S + s )* H * W + i] = 1;
                //data[(n * S + s )* H * W + i] = s;
                //data[(n * S + s )* H * W + i] = n + (i % W + 1);
                //data[(n * S + s )* H * W + i] = n + (i / W + 1);
                //data[(n * S + s )* H * W + i] = (i % W);

            }
        }
    }

    FillerParameter offset_filler_param;
    offset_filler_param.set_min(2);
    offset_filler_param.set_max(kernel_size-2);
    UniformFiller<float> offset_filler(offset_filler_param);

    //offset_filler.Fill(&blob_offsets);
    const_zero_filer.Fill(&blob_offsets_x);
    const_zero_filer.Fill(&blob_offsets_y);

    const_zero_float_filer.Fill(&blob_offsets_float_x);
    const_zero_float_filer.Fill(&blob_offsets_float_y);

    const_zero_float_filer.Fill(&blob_output);

    float* output = Caffe::mode() == Caffe::CPU ? blob_output.mutable_cpu_data() : blob_output.mutable_gpu_data();

    LayerParameter layer_param;

    ConvolutionParameter* gauss_convolution_param = layer_param.mutable_convolution_param();

    gauss_convolution_param->add_kernel_size(kernel_size);
    gauss_convolution_param->add_stride(1);
    gauss_convolution_param->add_pad(kernel_size/2);

    gauss_convolution_param->add_number_gauss(G);
    gauss_convolution_param->add_number_gauss(1);

    gauss_convolution_param->set_num_output(F);

    //gauss_convolution_param->mutable_weight_filler()->set_type("constant");
    //gauss_convolution_param->mutable_weight_filler()->set_value(2);

    gauss_convolution_param->mutable_weight_filler()->set_type("gaussian");
    gauss_convolution_param->mutable_weight_filler()->set_std(0.1);

    gauss_convolution_param->mutable_bias_filler()->set_type("constant");
    gauss_convolution_param->mutable_bias_filler()->set_value(0);

    gauss_convolution_param->mutable_mu_filler()->set_type("constant");
    gauss_convolution_param->mutable_mu_filler()->set_value(0);

    gauss_convolution_param->mutable_sigma_filler()->set_type("constant");
    gauss_convolution_param->mutable_sigma_filler()->set_value(0.8);

    gauss_convolution_param->set_gmm_component_border_bound(100);
    gauss_convolution_param->set_gmm_sigma_lower_bound(0.5);

    gauss_convolution_param->set_gmm_weight_normalization(false);
    gauss_convolution_param->set_gmm_gauss_normalization(true);
    gauss_convolution_param->set_gmm_square_gauss_normalization(false);

    FastAproxGaussianConvLayer<float> layer(layer_param);


    LayerParameter cudnn_layer_param;

    ConvolutionParameter* convolution_param =
            cudnn_layer_param.mutable_convolution_param();

    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(1);
    convolution_param->add_pad(1);

    convolution_param->set_num_output(F);

    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_weight_filler()->set_std(0.1);

    shared_ptr<CuDNNConvolutionLayer<float> > cudnn_layer(new CuDNNConvolutionLayer<float>(cudnn_layer_param));


    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_cudnn;
    std::vector<Blob<float>* > blob_top_vec_cpu;


    blob_bottom_vec.push_back(&blob_input);

    blob_top_vec.push_back(&blob_output);
    blob_top_vec_cudnn.push_back(&blob_output_cudnn);
    blob_top_vec_cpu.push_back(&blob_output_cpu);

    cudnn_layer->SetUp(blob_bottom_vec, blob_top_vec_cudnn);
/*
    for (int i = 0; i < 1; ++i) {
        clock_t start_t = clock();
        cudnn_layer->Forward(blob_bottom_vec, blob_top_vec_cudnn);
        cudaDeviceSynchronize();
        clock_t end_t = clock();
        std::cout << "CuDNNConvolutionLayer forward pass in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
    }*/
    std::cout << std::endl;
    for (int ii = 0; ii < 1; ++ii) {

        layer.SetUp(blob_bottom_vec, blob_top_vec);

        // override offset data with zeros
        float* filter_offsets_float_mu1 = layer.param_buffer_mu1_->mutable_gpu_data();
        float* filter_offsets_float_mu2 = layer.param_buffer_mu2_->mutable_gpu_data();

        cudaMemset(filter_offsets_float_mu1, 0, layer.param_buffer_mu1_->count() * sizeof(float));
        cudaMemset(filter_offsets_float_mu2, 0, layer.param_buffer_mu2_->count() * sizeof(float));

        offset_filler.Fill(layer.param_buffer_mu1_.get());
        offset_filler.Fill(layer.param_buffer_mu2_.get());

        float* mu1_data = layer.param_buffer_mu1_->mutable_cpu_data();
        float* mu2_data = layer.param_buffer_mu2_->mutable_cpu_data();
        float* w_data = layer.param_buffer_w_->mutable_cpu_data();


        for (int s = 0; s < S; s++) {
            for (int g = 0; g < G; g++) {
                for (int f = 0; f < F; f++) {
                    //w_data[OFFSET(0,s,g,f, 1, S,G,F)] = floor(w_data[OFFSET(0,s,g,f, 1, S,G,F)] * 100) / 100.0f ;
                    //w_data[OFFSET(0,s,g,f, 1, S,G,F)] = s;
                    //w_data[OFFSET(0,s,g,f, 1, S,G,F)] = 1;
                    //mu1_data[OFFSET(0,s,g,f, 1, S,G,F)] = -2.2 + ((f+1)*(1+s)*(g+1)) % 5 ;
                    //mu2_data[OFFSET(0,s,g,f, 1, S,G,F)] = 3.1 - ((f+1)*(1+s)*(g+1)) % 5 ;
                    //mu1_data[OFFSET(0,s,g,f, 1, S,G,F)] = kernel_size/2+1;
                    //mu2_data[OFFSET(0,s,g,f, 1, S,G,F)] = kernel_size/2;
                }
            }
        }


        layer.Forward_cpu(blob_bottom_vec, blob_top_vec_cpu);

        layer.Forward_gpu(blob_bottom_vec, blob_top_vec);

        float* output_cpu = blob_top_vec_cpu[0]->mutable_cpu_data();
        float* output_gpu = blob_top_vec[0]->mutable_cpu_data();


        // verify data - since we use 1 for input and wights and 0 for offsets we should get S as output value for all

        const bool compare_by_cpu = true;
        int found_invalid = 0;
        //double valid_value = S *G;
        double valid_value = (S-1) * (S)/2 * G;
        double diff = 0;
        double max_diff = 0;

        for (int n = 0; n < N; ++n){
            for (int f = 0; f < F; ++f) {
                for (int i = 0; i < H * W; ++i) {
                    int index = (n * F + f )* H * W + i;
                    float val = output_gpu[index];
                    float GT_VALUE = (n + i % W +1 )*G*S;  // for data[] = n + (i % W + 1)
                    //float GT_VALUE = (n + i / W +1 )*G*S;  // for data[] = n + (i  / W + 1)
                    //float GT_VALUE = (i % W  ) *G*S;       // for data[] = s
                    //float GT_VALUE = n + (i % W + 1);      // for just copy

                    if (compare_by_cpu) {
                        GT_VALUE = output_cpu[index];
                    }

                    // interpolation at the right edge excludes one pixel so ignore those pixels

                    if (std::abs(val - GT_VALUE) / GT_VALUE > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && i % W < (W -1)) {
                        if (found_invalid < 10)
                            printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index,  n, f, i / W, i % W, GT_VALUE);
                        found_invalid++;

                        double current_diff = std::abs(val - GT_VALUE) / GT_VALUE;

                        diff += current_diff;

                        max_diff = std::max(max_diff, current_diff);
                    }
                    //if (i % W == 0)
                    //    std::cout << std::endl;
                    //std::cout << val << " ";
                    //std::cout << GT_VALUE << " ";
                }
                //std::cout << std::endl;
            }
        }
        //std::cout << std::endl;
        diff /= found_invalid;

        if (found_invalid > 0)
            printf("found num of invalid output vals: %d/%d with mean diff val %f and max diff val %f\n",found_invalid, blob_output.count(), diff, max_diff);
    }
}

TYPED_TEST(GaussConvolutionLayerTest, DebugFastGaussBackwardMultiSubfeatures) {


    typedef typename TypeParam::Dtype Dtype;

    Caffe::SetDevice(0);

    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;
    // evaluate size settings

    const int N = 128;
    const int F = 32;
    const int S = 1;
    const int G = 2;
    const int W = 64;
    const int H = 32;

    // number of Guassian learning parameters we have (w,mu1,mu2,sigma)
    // for each parameter we need convolution of input data with specific kernel
    const int K = 3;
    const bool use_interpolation = true;
    const bool ignore_edge_gradients = true; // for cpu/gpu compatability

    const int kernel_size = 9;

    Blob<float> blob_input(N,S,H,W);
    //Blob<float> blob_input(N,S * K,H,W);
    Blob<float> blob_error(N,F,H,W);
    Blob<float> blob_error_gt(N,F,H,W);
    Blob<int> blob_offsets_x(1, S, G, F);
    Blob<int> blob_offsets_y(1, S, G, F);
    Blob<float> blob_offsets_float_x(1, S, G, F);
    Blob<float> blob_offsets_float_y(1, S, G, F);
    Blob<float> blob_weights(1, S, G, F);
    Blob<float> blob_output(K, S, G, F);
    Blob<float> blob_output_cpu(K, S, G, F);
    Blob<float> blob_output_gt(K, S, G, F);

    Blob<float> blob_output_error_cpu(N,S,H,W);
    Blob<float> blob_output_error_gt(N,S,H,W);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_value(1);
    GaussianFiller<float> input_filler(rand_filler_param);

    input_filler.Fill(&blob_input);
    input_filler.Fill(&blob_weights);
    input_filler.Fill(&blob_error);
    caffe_rng_gaussian<float>(blob_error.count(), float(0), float(0.1), blob_error.mutable_cpu_diff());


    //const_one_filer.Fill(&blob_input);
    //const_one_filer.Fill(&blob_error);
    //const_one_filer.Fill(&blob_weights);

    float* data = blob_input.mutable_cpu_data();
    //for (int n = 0; n < N*S*K; ++n){
    for (int n = 0; n < N; ++n){
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < H * W; ++i) {
                //data[n*H*W + i] = i % W + 1;
                //data[( n * S  +   s  )* H * W + i] = (s ) +n*4;
                //data[n*H*W + i] = 1;
            }
        }
    }

    float* error = blob_error.mutable_cpu_diff();
    for (int n = 0; n < N*F; ++n){
        for (int i = 0; i < H*W; ++i) {
            //error[n*H*W + i] = i % W + 2;
            //error[n*H*W + i] = 1;
            //error[n*H*W + i] = (n+1);
        }
    }

    float* error_gt = blob_error_gt.mutable_cpu_diff();
    for (int n = 0; n < N*F; ++n){
        for (int i = 0; i < H*W; ++i) {
            // set error values at the bottom/right borders to zero for compatability with FastGPU version
            error_gt[n*H*W + i] = i % W < W -1 && i / W < H-1 ? error[n*H*W + i] : 0;
        }
    }


    FillerParameter offset_filler_param;
    offset_filler_param.set_min(4);
    offset_filler_param.set_max(kernel_size - 4);

    UniformFiller<float> offset_filler(offset_filler_param);

    //offset_filler.Fill(&blob_offsets);
    const_zero_filer.Fill(&blob_offsets_x);
    const_zero_filer.Fill(&blob_offsets_y);

    const_zero_float_filer.Fill(&blob_offsets_float_x);
    const_zero_float_filer.Fill(&blob_offsets_float_y);

    const_zero_float_filer.Fill(&blob_output);

    LayerParameter layer_param;
    ConvolutionParameter* gauss_convolution_param = layer_param.mutable_convolution_param();

    gauss_convolution_param->add_kernel_size(kernel_size);
    gauss_convolution_param->add_stride(1);
    gauss_convolution_param->add_pad(kernel_size/2);

    gauss_convolution_param->add_number_gauss(G);
    gauss_convolution_param->add_number_gauss(1);

    gauss_convolution_param->set_num_output(F);

    //gauss_convolution_param->mutable_weight_filler()->set_type("constant");
    //gauss_convolution_param->mutable_weight_filler()->set_value(1);

    gauss_convolution_param->mutable_weight_filler()->set_type("gaussian");
    gauss_convolution_param->mutable_weight_filler()->set_std(0.1);

    gauss_convolution_param->mutable_bias_filler()->set_type("constant");
    gauss_convolution_param->mutable_bias_filler()->set_value(0);

    gauss_convolution_param->mutable_mu_filler()->set_type("constant");
    gauss_convolution_param->mutable_mu_filler()->set_value(0);

    gauss_convolution_param->mutable_sigma_filler()->set_type("constant");
    gauss_convolution_param->mutable_sigma_filler()->set_value(0.8);

    gauss_convolution_param->set_gmm_component_border_bound(0);
    gauss_convolution_param->set_gmm_sigma_lower_bound(0.5);

    gauss_convolution_param->set_gmm_weight_normalization(false);
    gauss_convolution_param->set_gmm_gauss_normalization(true);
    gauss_convolution_param->set_gmm_square_gauss_normalization(false);

    FastAproxGaussianConvLayer<float> layer(layer_param);
    CuDNNGaussianConvLayer<float> layer_gt(layer_param);

    layer.ignore_edge_gradients_ = ignore_edge_gradients;

    LayerParameter cudnn_layer_param;

    ConvolutionParameter* convolution_param =
            cudnn_layer_param.mutable_convolution_param();

    convolution_param->add_kernel_size(7);
    convolution_param->add_stride(1);
    convolution_param->add_pad(2);

    convolution_param->set_num_output(F);

    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_weight_filler()->set_std(0.01);

    shared_ptr<CuDNNConvolutionLayer<float> > cudnn_layer(new CuDNNConvolutionLayer<float>(cudnn_layer_param));

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;
    std::vector<Blob<float>* > blob_top_vec_gt;
    std::vector<bool > propagate_down;
    propagate_down.push_back(true);

    blob_bottom_vec.push_back(&blob_input);
    blob_top_vec.push_back(&blob_error);
    blob_top_vec_gt.push_back(&blob_error_gt);
    /*{
        std::vector<Blob<float>* > blob_bottom_vec_cudnn;
        std::vector<Blob<float>* > blob_top_vec_cudnn;

        Blob<float> blob_input(N,S,H,W);
        Blob<float> blob_error(N,F,H,W);

        blob_bottom_vec_cudnn.push_back(&blob_input);
        blob_top_vec_cudnn.push_back(&blob_error);

        cudnn_layer->SetUp(blob_bottom_vec_cudnn, blob_top_vec_cudnn);
        cudaDeviceSynchronize();

        for (int i = 0 ; i < 1; ++i) {
            clock_t start_t = clock();
            cudnn_layer->Backward(blob_top_vec_cudnn, propagate_down, blob_bottom_vec_cudnn);
            cudaDeviceSynchronize();
            clock_t end_t = clock();
            std::cout << "CuDNNConvolutionLayer backward pass in " << (((float) (end_t - start_t)) / CLOCKS_PER_SEC) << std::endl;
        }
    }
    std::cout << std::endl;*/
    for (int ii = 0; ii < 1; ++ii) {

        layer.SetUp(blob_bottom_vec, blob_top_vec);
        layer_gt.SetUp(blob_bottom_vec, blob_top_vec);

        // override offset data with zeros
        float *filter_offsets_float_mu1 = layer.param_buffer_mu1_->mutable_gpu_data();
        float *filter_offsets_float_mu2 = layer.param_buffer_mu2_->mutable_gpu_data();


        cudaMemset(filter_offsets_float_mu1, 0, layer.param_buffer_mu1_->count() * sizeof(float));
        cudaMemset(filter_offsets_float_mu2, 0, layer.param_buffer_mu2_->count() * sizeof(float));

        offset_filler.Fill(layer.param_buffer_mu1_.get());
        offset_filler.Fill(layer.param_buffer_mu2_.get());

        float *w_data = layer.param_buffer_w_->mutable_cpu_data();
        float *mu1_data = layer.param_buffer_mu1_->mutable_cpu_data();
        float *mu2_data = layer.param_buffer_mu2_->mutable_cpu_data();
        float *sigma_data = layer.param_buffer_sigma_->mutable_cpu_data();

        float *filter_offsets_float_mu1_gt = layer_gt.param_buffer_mu1_->mutable_gpu_data();
        float *filter_offsets_float_mu2_gt = layer_gt.param_buffer_mu2_->mutable_gpu_data();

        cudaMemset(filter_offsets_float_mu1_gt, 0, layer_gt.param_buffer_mu1_->count() * sizeof(float));
        cudaMemset(filter_offsets_float_mu2_gt, 0, layer_gt.param_buffer_mu2_->count() * sizeof(float));

        float *w_data_gt = layer_gt.param_buffer_w_->mutable_cpu_data();
        float *mu1_data_gt = layer_gt.param_buffer_mu1_->mutable_cpu_data();
        float *mu2_data_gt = layer_gt.param_buffer_mu2_->mutable_cpu_data();
        float *sigma_data_gt = layer_gt.param_buffer_sigma_->mutable_cpu_data();

        for (int s = 0; s < S; s++) {
            for (int g = 0; g < G; g++) {
                for (int f = 0; f < F; f++) {
                    //w_data[OFFSET(0,s,g,f, 1, S,G,F)] = 1;
                    //w_data[OFFSET(0,s,g,f, 1, S,G,F)] = s;
                    //w_data[OFFSET(0,s,g,f, 1, S,G,F)] = (s+1)*(f+1)*0.01;
                    //mu1_data[OFFSET(0, s, g, f, 1, S, G, F)] = -2.2 + ((f+1)*(1+s)*(g+1)) % 5 ;
                    //mu1_data[OFFSET(0, s, g, f, 1, S, G, F)] = kernel_size/2;
                    //mu2_data[OFFSET(0, s, g, f, 1, S, G, F)] = kernel_size/2;
                    /*
                    // discretize the weights
                    mu1_data[OFFSET(0, s, g, f, 1, S, G, F)] = floor(mu1_data[OFFSET(0, s, g, f, 1, S, G, F)]);
                    mu2_data[OFFSET(0, s, g, f, 1, S, G, F)] = floor(mu2_data[OFFSET(0, s, g, f, 1, S, G, F)]);

                    w_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = w_data[OFFSET(0,s,g,f, 1, S,G,F)];
                    // and set the same values for groundtruth layer
                    mu1_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = mu1_data[OFFSET(0, s, g, f, 1, S, G, F)];
                    mu2_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = mu2_data[OFFSET(0, s, g, f, 1, S, G, F)];
                    //mu1_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = kernel_size/2;
                    //mu2_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = kernel_size/2;

                    // set fixed sigma for all components in groundtruth layer
                    sigma_data_gt[OFFSET(0, s, g, f, 1, S, G, F)] = sigma_data[OFFSET(0, s, g, f, 1, S, G, F)];*/
                }
            }
        }

        /*layer_gt.Backward_gpu(blob_top_vec_gt, propagate_down, blob_bottom_vec);

        // save backproped error values
        float *backprop_error_gt = blob_output_error_gt.mutable_cpu_data();
        {
            caffe_copy(blob_output_error_gt.count(), blob_bottom_vec[0]->cpu_diff(), backprop_error_gt);
        }

        // save accumulated gradient values
        float *gradients_gt = blob_output_gt.mutable_cpu_data();
        {
            int num_params = layer_gt.param_buffer_w_->count();
            if (K > 0) caffe_copy(num_params, layer_gt.param_buffer_w_->cpu_diff(), gradients_gt + 0 * num_params );
            if (K > 1) caffe_copy(num_params, layer_gt.param_buffer_mu1_->cpu_diff(), gradients_gt + 1 * num_params );
            if (K > 2) caffe_copy(num_params, layer_gt.param_buffer_mu2_->cpu_diff(), gradients_gt + 2 * num_params );
            if (K > 3) caffe_copy(num_params, layer_gt.param_buffer_sigma_->cpu_diff(), gradients_gt + 3 * num_params);
        }
*/
        caffe_gpu_set(layer.param_buffer_w_->count(), (Dtype)0, (Dtype*)layer.param_buffer_w_->mutable_gpu_diff());
        caffe_gpu_set(layer.param_buffer_mu1_->count(), (Dtype)0, (Dtype*)layer.param_buffer_mu1_->mutable_gpu_diff());
        caffe_gpu_set(layer.param_buffer_mu2_->count(), (Dtype)0, (Dtype*)layer.param_buffer_mu2_->mutable_gpu_diff());
        caffe_gpu_set(layer.param_buffer_sigma_->count(), (Dtype)0, (Dtype*)layer.param_buffer_sigma_->mutable_gpu_diff());

        layer.Backward_cpu(blob_top_vec, propagate_down, blob_bottom_vec);

        // save backproped error values
        float *backprop_error_cpu = blob_output_error_cpu.mutable_cpu_data();
        {
            caffe_copy(blob_output_error_cpu.count(), blob_bottom_vec[0]->cpu_diff(), backprop_error_cpu);
        }

        // save accumulated gradient values
        float *gradients_cpu = blob_output_cpu.mutable_cpu_data();
        {
            int num_params = layer.param_buffer_w_->count();
            if (K > 0) caffe_copy(num_params, layer.param_buffer_w_->cpu_diff(), gradients_cpu + 0 * num_params );
            if (K > 1) caffe_copy(num_params, layer.param_buffer_mu1_->cpu_diff(), gradients_cpu + 1 * num_params );
            if (K > 2) caffe_copy(num_params, layer.param_buffer_mu2_->cpu_diff(), gradients_cpu + 2 * num_params );
            if (K > 3) caffe_copy(num_params, layer.param_buffer_sigma_->cpu_diff(), gradients_cpu + 3 * num_params);
        }

        // reset values for GPU run
        caffe_gpu_set(layer.param_buffer_w_->count(), (Dtype)0, (Dtype*)layer.param_buffer_w_->mutable_gpu_diff());
        caffe_gpu_set(layer.param_buffer_mu1_->count(), (Dtype)0, (Dtype*)layer.param_buffer_mu1_->mutable_gpu_diff());
        caffe_gpu_set(layer.param_buffer_mu2_->count(), (Dtype)0, (Dtype*)layer.param_buffer_mu2_->mutable_gpu_diff());
        caffe_gpu_set(layer.param_buffer_sigma_->count(), (Dtype)0, (Dtype*)layer.param_buffer_sigma_->mutable_gpu_diff());


        layer.Backward_gpu(blob_top_vec, propagate_down, blob_bottom_vec);

        // get ptr to backproped error on gpu
        const float *backprop_error_gpu = blob_bottom_vec[0]->cpu_diff();

        // save accumulated gradient values
        float *gradients_gpu_g = blob_output.mutable_gpu_data();

        int num_params = layer.param_buffer_w_->count();
        if (K > 0) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_w_->gpu_diff(), gradients_gpu_g + 0 * num_params );
        if (K > 1) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_mu1_->gpu_diff(), gradients_gpu_g + 1 * num_params );
        if (K > 2) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_mu2_->gpu_diff(), gradients_gpu_g + 2 * num_params );
        if (K > 3) caffe_gpu_memcpy(num_params * sizeof(float), layer.param_buffer_sigma_->gpu_diff(), gradients_gpu_g + 3 * num_params);

        // get gradients from GPU, but read them on cpu
        float *gradients_gpu = blob_output.mutable_cpu_data();

        {
            // verify accumulated gradients
            int found_invalid = 0;

            //double WH = (double)W*(double)H;
            double WH = (double) W;

            double GT_VALUE = (double)N*W*H;
            //double GT_VALUE = N*((WH)*((WH-1))/2); // input x error <== [1..N] x [1 1 .. 1]
            //double GT_VALUE = H*((WH)*((WH-1))/2); // input x error <== [1..N] x [1 1 .. 1]
            //double GT_VALUE = N*(((WH-1)*((WH-1)+1)*(2*(WH-1)+1))/6);
            //double GT_VALUE = H * N * (((WH - 1) * ((WH - 1) + 1) * (2 * (WH - 1) + 1)) / 6); // for i % W  using in data and error !!

            double diff = 0;
            double max_diff = 0;
            for (int k = 0; k < K; ++k) {
                std::cout << "k=" << k << std::endl;
                for (int s = 0; s < S; ++s) {
                    for (int g = 0; g < G; ++g) {
                        for (int f = 0; f < F; ++f) {
                            int idx = OFFSET(k,s,g,f, K, S,  G, F);
                            float val = gradients_gpu[idx];
                            float GT_VALUE = gradients_cpu[idx];
                            //float GT_VALUE = gradients_gt[idx];

                            if (std::abs(val - GT_VALUE) / GT_VALUE > 1e-4 && std::abs(val - GT_VALUE) > 1e-7) {
                                if (found_invalid < 10)
                                    printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, idx, k, s,g,f, GT_VALUE);
                                found_invalid++;
                                double current_diff = std::abs(val - GT_VALUE) / GT_VALUE;

                                diff += current_diff;

                                max_diff = std::max(max_diff, current_diff);

                            }
                            //std::cout << val << " ";
                            //std::cout << GT_VALUE << " ";
                        }
                        //std::cout << std::endl;
                    }
                }
            }

            diff /= found_invalid;

            if (found_invalid > 0)
                printf("found num of invalid accumulated gradient vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid, K*S*G*F, diff, max_diff);

        }
        {
            // verify accumulated gradients
            int found_invalid = 0;

            double diff = 0;
            double max_diff = 0;
            for (int n = 0; n < N; ++n) {
                for (int s = 0; s < S; ++s) {
                    for (int i = 0; i < H * W; ++i) {
                        int index = (n * S + s )* H * W + i;
                        float val = backprop_error_gpu[index];
                        float GT_VALUE = backprop_error_cpu[index];
                        //float GT_VALUE = backprop_error_gt[index];

                        if (std::abs(val - GT_VALUE) / GT_VALUE > 1e-4 && std::abs(val - GT_VALUE) > 1e-7 && i % W < (W -1)) {
                            if (found_invalid < 10)
                                printf("found invalid output (%f) at loc (%d=%d,%d,%d,%d) - should be %f\n", val, index, n, s, i / W, i % W, GT_VALUE);
                            found_invalid++;
                            diff += std::abs(val - GT_VALUE) / GT_VALUE;

                            double current_diff = std::abs(val - GT_VALUE) / GT_VALUE;

                            diff += current_diff;

                            max_diff = std::max(max_diff, current_diff);


                        }
                        //if (i % W == 0)
                        //    std::cout << std::endl;
                        //std::cout << val << " ";
                        //std::cout << GT_VALUE << " ";
                    }
                    //std::cout << std::endl;
                }
            }

            diff /= found_invalid;

            if (found_invalid > 0)
                printf("found num of invalid backproped-error vals: %d/%d with mean diff val %f and max diff val %f\n", found_invalid, N*S*H*W, diff, max_diff);
        }
    }
}

#ifdef USE_CUDNN

#endif

}  // namespace caffe
