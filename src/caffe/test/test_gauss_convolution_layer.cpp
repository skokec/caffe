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

TYPED_TEST(GaussConvolutionLayerTest, TestFastGaussConvolution) {


    Caffe::SetDevice(0);

    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;
    // evaluate size settings
    const int N = 128;
    const int F = 128;
    const int S = 128;
    const int G = 2;
    const int W = 64;
    const int H = 64;
    const bool use_interpolation = true;

    // debug print version
/*
    const int N = 1;
    const int F = 32;
    const int S = 32;
    const int G = 1;
    const int W = 32;
    const int H = 16;
*/

    Blob<float> blob_input(N,S,H,W);
    Blob<int> blob_offsets_x(1, S, G, F);
    Blob<int> blob_offsets_y(1, S, G, F);
    Blob<float> blob_offsets_float_x(1, S, G, F);
    Blob<float> blob_offsets_float_y(1, S, G, F);
    Blob<float> blob_weights(1, S, G, F);
    Blob<float> blob_output(N,F,H,W);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_value(0.1);
    GaussianFiller<float> input_filler(rand_filler_param);

    //input_filler.Fill(&blob_input);
    //input_filler.Fill(&blob_weights);

    const_one_filer.Fill(&blob_input);
    const_one_filer.Fill(&blob_weights);


    float* data = blob_input.mutable_cpu_data();
    for (int n = 0; n < N; ++n){
        for (int s = 0; s < S; ++s) {
            for (int i = 0; i < H * W; ++i) {
                //data[(n * S + s )* H * W + i] = s;
                data[(n * S + s )* H * W + i] = n + (i % W + 1);
                //data[(n * S + s )* H * W + i] = (i % W);

            }
        }
    }

    FillerParameter offset_filler_param;
    offset_filler_param.set_max(2);
    offset_filler_param.set_min(0);
    UniformFiller<float> offset_filler(offset_filler_param);

    //offset_filler.Fill(&blob_offsets);
    const_zero_filer.Fill(&blob_offsets_x);
    const_zero_filer.Fill(&blob_offsets_y);

    const_zero_float_filer.Fill(&blob_offsets_float_x);
    const_zero_float_filer.Fill(&blob_offsets_float_y);

    const_zero_float_filer.Fill(&blob_output);

    const float* filtered_images = Caffe::mode() == Caffe::CPU ? blob_input.cpu_data() : blob_input.gpu_data();
    const int* filter_offsets_x = Caffe::mode() == Caffe::CPU ? blob_offsets_x.cpu_data() : blob_offsets_x.gpu_data();
    const int* filter_offsets_y = Caffe::mode() == Caffe::CPU ? blob_offsets_y.cpu_data() : blob_offsets_y.gpu_data();
    const float* filter_offsets_float_x = Caffe::mode() == Caffe::CPU ? blob_offsets_float_x.cpu_data() : blob_offsets_float_x.gpu_data();
    const float* filter_offsets_float_y = Caffe::mode() == Caffe::CPU ? blob_offsets_float_y.cpu_data() : blob_offsets_float_y.gpu_data();
    const float* filter_weights = Caffe::mode() == Caffe::CPU ? blob_weights.cpu_data() : blob_weights.gpu_data();
    float* output = Caffe::mode() == Caffe::CPU ? blob_output.mutable_cpu_data() : blob_output.mutable_gpu_data();


    LayerParameter layer_param;
    FastAproxGaussianConvLayer<float> layer(layer_param);

    LayerParameter cudnn_layer_param;

    ConvolutionParameter* convolution_param =
            cudnn_layer_param.mutable_convolution_param();

    convolution_param->add_kernel_size(3);
    convolution_param->add_stride(1);
    convolution_param->add_pad(1);

    convolution_param->set_num_output(F);

    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_weight_filler()->set_std(0.01);

    std::cout << "input size:" << std::endl;
    std::cout << "blob_input: " << blob_input.count()  << std::endl;
    std::cout << "blob_offsets_x: " << blob_offsets_x.count()  << std::endl;
    std::cout << "blob_offsets_y: " << blob_offsets_y.count()  << std::endl;
    std::cout << "blob_weights: " << blob_weights.count()  << std::endl;
    std::cout << "blob_output: " << blob_output.count() << std::endl << std::endl;

    shared_ptr<CuDNNConvolutionLayer<float> > cudnn_layer(new CuDNNConvolutionLayer<float>(cudnn_layer_param));

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;

    blob_bottom_vec.push_back(&blob_input);
    blob_top_vec.push_back(&blob_output);

    cudnn_layer->SetUp(blob_bottom_vec, blob_top_vec);

    for (int i = 0; i < 30; ++i) {
        clock_t start_t = clock();
        cudnn_layer->Forward(blob_bottom_vec, blob_top_vec);
        cudaDeviceSynchronize();
        clock_t end_t = clock();
        std::cout << "CuDNNConvolutionLayer forward pass in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
    }
    std::cout << std::endl;
    for (int ii = 0; ii < 1; ++ii) {

	    if (Caffe::mode() == Caffe::CPU)
		    layer.test_kernel_cpu(filtered_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, N, S, F, G, W, H, 5, 5, use_interpolation);
	    else
		    layer.test_kernel_gpu(filtered_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, N, S, F, G, W, H, 5, 5, use_interpolation);


        float* output_c = blob_output.mutable_cpu_data();

        // verify data - since we use 1 for input and wights and 0 for offsets we should get S as output value for all

        int found_invalid = 0;
        //double valid_value = S *G;
        double valid_value = (S-1) * (S)/2 * G;

        for (int n = 0; n < N; ++n){
            for (int f = 0; f < F; ++f) {
                for (int i = 0; i < H * W; ++i) {
                    int index = (n * F + f )* H * W + i;
                    float val = output_c[index];
                    float valid_value = (n + i % W +1 )*G*S;
                    //float valid_value = (i % W  ) *G*S;
                    if (val != valid_value) {
                        if (found_invalid < 10)
                            printf("found invalid output (%f) at loc (%d) - should be %f\n", val, index, valid_value);
                        found_invalid++;
                    }
                    /*if (i % W == 0)
                        std::cout << std::endl;
                    std::cout << val << " ";*/
                }
            }
        }
        std::cout << std::endl;
/*
        for (int jj = 0; jj < blob_output.count(); ++jj) {
            if (output_c[jj] != valid_value) {
                if (found_invalid < 10)
                    printf("found invalid output (%f) at loc (%d) - should be %f\n", output_c[jj], jj, valid_value);
                found_invalid++;
            }
        }
*/
        if (found_invalid > 0)
            printf("found num of invalid output vals: %d/%d\n",found_invalid, blob_output.count());
    }
}

TYPED_TEST(GaussConvolutionLayerTest, TestFastGaussBackward) {

    Caffe::SetDevice(0);


    typedef typename TypeParam::Dtype Dtype;


    if (Caffe::mode() == Caffe::CPU)
        return;

    if (sizeof(Dtype) > 4)
        return;
    // evaluate size settings
    /*

    const int N = 128;
    const int F = 128;
    const int S = 128;
    const int G = 2;
    const int W = 64;
    const int H = 64;
    */
    // debug print version

    const int N = 128;
    const int F = 128;
    const int S = 128;
    const int G = 2;
    const int W = 64;
    const int H = 64;

        /*
    const int N = 128;
    const int F = 128;
    const int S = 128;
    const int G = 1;
    const int W = 64;
    const int H = 64;
*/
    Blob<float> blob_input(N,S,H,W);
    Blob<float> blob_error(N,F,H,W);
    Blob<int> blob_offsets_x(1, S, G, F);
    Blob<int> blob_offsets_y(1, S, G, F);
    Blob<float> blob_offsets_float_x(1, S, G, F);
    Blob<float> blob_offsets_float_y(1, S, G, F);
    Blob<float> blob_weights(1, S, G, F);
    Blob<float> blob_output(1, S, G, F);

    FillerParameter const_one_filler_param;
    const_one_filler_param.set_value(1);
    ConstantFiller<float> const_one_filer(const_one_filler_param);

    FillerParameter const_zero_filler_param;
    const_zero_filler_param.set_value(0);
    ConstantFiller<int> const_zero_filer(const_zero_filler_param);
    ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

    FillerParameter rand_filler_param;
    rand_filler_param.set_value(0.1);
    GaussianFiller<float> input_filler(rand_filler_param);

    //input_filler.Fill(&blob_input);
    //input_filler.Fill(&blob_weights);

    const_one_filer.Fill(&blob_input);
    const_one_filer.Fill(&blob_error);
    const_one_filer.Fill(&blob_weights);

    float* data = blob_input.mutable_cpu_data();
    for (int n = 0; n < N*S; ++n){
        for (int i = 0; i < H*W; ++i)
            data[n*H*W + i] = i;
    }


    FillerParameter offset_filler_param;
    offset_filler_param.set_max(2);
    offset_filler_param.set_min(0);
    UniformFiller<float> offset_filler(offset_filler_param);

    //offset_filler.Fill(&blob_offsets);
    const_zero_filer.Fill(&blob_offsets_x);
    const_zero_filer.Fill(&blob_offsets_y);

    const_zero_float_filer.Fill(&blob_offsets_float_x);
    const_zero_float_filer.Fill(&blob_offsets_float_y);

    const_zero_float_filer.Fill(&blob_output);

    const float* filtered_images = Caffe::mode() == Caffe::CPU ? blob_input.cpu_data() : blob_input.gpu_data();
    const float* error_images = Caffe::mode() == Caffe::CPU ? blob_error.cpu_data() : blob_error.gpu_data();
    const int* filter_offsets_x = Caffe::mode() == Caffe::CPU ? blob_offsets_x.cpu_data() : blob_offsets_x.gpu_data();
    const int* filter_offsets_y = Caffe::mode() == Caffe::CPU ? blob_offsets_y.cpu_data() : blob_offsets_y.gpu_data();
    const float* filter_offsets_float_x = Caffe::mode() == Caffe::CPU ? blob_offsets_float_x.cpu_data() : blob_offsets_float_x.gpu_data();
    const float* filter_offsets_float_y = Caffe::mode() == Caffe::CPU ? blob_offsets_float_y.cpu_data() : blob_offsets_float_y.gpu_data();
    const float* filter_weights = Caffe::mode() == Caffe::CPU ? blob_weights.cpu_data() : blob_weights.gpu_data();
    float* output = Caffe::mode() == Caffe::CPU ? blob_output.mutable_cpu_data() : blob_output.mutable_gpu_data();


    LayerParameter layer_param;
    FastAproxGaussianConvLayer<float> layer(layer_param);

    LayerParameter cudnn_layer_param;

    ConvolutionParameter* convolution_param =
         cudnn_layer_param.mutable_convolution_param();

    convolution_param->add_kernel_size(5);
    convolution_param->add_stride(1);
    convolution_param->add_pad(2);

    convolution_param->set_num_output(F);

    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_weight_filler()->set_std(0.01);

    std::cout << "input size:" << std::endl;
    std::cout << "blob_input: " << blob_input.count()  << std::endl;
    std::cout << "blob_offsets_x: " << blob_offsets_x.count()  << std::endl;
    std::cout << "blob_offsets_y: " << blob_offsets_y.count()  << std::endl;
    std::cout << "blob_weights: " << blob_weights.count()  << std::endl;
    std::cout << "blob_output: " << blob_output.count() << std::endl << std::endl;

    shared_ptr<CuDNNConvolutionLayer<float> > cudnn_layer(new CuDNNConvolutionLayer<float>(cudnn_layer_param));

    std::vector<Blob<float>* > blob_bottom_vec;
    std::vector<Blob<float>* > blob_top_vec;

    blob_bottom_vec.push_back(&blob_input);
    blob_top_vec.push_back(&blob_output);

    /*
    cudnn_layer->SetUp(blob_bottom_vec, blob_top_vec);

    clock_t start_t = clock();
    cudnn_layer->Forward(blob_bottom_vec, blob_top_vec);
    cudaDeviceSynchronize();
    clock_t end_t = clock();
    std::cout << "CuDNNConvolutionLayer forward pass in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl << std::endl;
    */
    for (int ii = 0; ii < 1; ++ii) {

         if (Caffe::mode() == Caffe::GPU)
             layer.test_backward_kernel_gpu(filtered_images, error_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, N, S, F, G, W, H, 5, 5);

         float* output_c = blob_output.mutable_cpu_data();

         // verify data - since we use 1 for input and wights and 0 for offsets we should get N*W*H as output value for all

         int found_invalid = 0;

        //int GT_VALUE = N*W*H;
        int GT_VALUE = N*((W*H)*((W*H-1))/2);
         for (int jj = 0; jj < blob_output.count(); ++jj) {
             if (output_c[jj] != GT_VALUE) {
                 if (found_invalid < 10)
                     printf("found invalid output (%f) at loc (%d) - should be %d\n", output_c[jj], jj, GT_VALUE);
                 found_invalid++;
             }
         }

         if (found_invalid > 0)
             printf("found num of invalid output vals: %d/%d\n",found_invalid, blob_output.count());
    }
}
    TYPED_TEST(GaussConvolutionLayerTest, TestFastGaussBackwardMultiSubfeatures) {


        typedef typename TypeParam::Dtype Dtype;

        Caffe::SetDevice(0);

        if (Caffe::mode() == Caffe::CPU)
            return;

        if (sizeof(Dtype) > 4)
            return;
        // evaluate size settings
        /*

        const int N = 128;
        const int F = 128;
        const int S = 128;
        const int G = 2;
        const int W = 64;
        const int H = 64;
        */
        // debug print version

        const int N = 128;
        const int F = 128;
        const int S = 128;
        const int G = 2;
        const int W = 64;
        const int H = 64;

        /*
    const int N = 128;
    const int F = 128;
    const int S = 128;
    const int G = 1;
    const int W = 64;
    const int H = 64;
*/
        // number of Guassian learning parameters we have (w,mu1,mu2,sigma)
        // for each parameter we need convolution of input data with specific kernel
        const int K = 4;
        const bool use_interpolation = true;

        Blob<float> blob_input(N,S * K,H,W);
        Blob<float> blob_error(N,F,H,W);
        Blob<int> blob_offsets_x(1, S, G, F);
        Blob<int> blob_offsets_y(1, S, G, F);
        Blob<float> blob_offsets_float_x(1, S, G, F);
        Blob<float> blob_offsets_float_y(1, S, G, F);
        Blob<float> blob_weights(1, S, G, F);
        Blob<float> blob_output(K, S, G, F);

        FillerParameter const_one_filler_param;
        const_one_filler_param.set_value(1);
        ConstantFiller<float> const_one_filer(const_one_filler_param);

        FillerParameter const_zero_filler_param;
        const_zero_filler_param.set_value(0);
        ConstantFiller<int> const_zero_filer(const_zero_filler_param);
        ConstantFiller<float> const_zero_float_filer(const_zero_filler_param);

        FillerParameter rand_filler_param;
        rand_filler_param.set_value(0.1);
        GaussianFiller<float> input_filler(rand_filler_param);

        //input_filler.Fill(&blob_input);
        //input_filler.Fill(&blob_weights);

        const_one_filer.Fill(&blob_input);
        const_one_filer.Fill(&blob_error);
        const_one_filer.Fill(&blob_weights);

        float* data = blob_input.mutable_cpu_data();
        for (int n = 0; n < N*S*K; ++n){
            for (int i = 0; i < H*W; ++i) {
                data[n*H*W + i] = i % W ;
            }
        }

        float* error = blob_error.mutable_cpu_data();
        for (int n = 0; n < N*F; ++n){
            for (int i = 0; i < H*W; ++i) {
                error[n*H*W + i] = i % W;
            }
        }


        FillerParameter offset_filler_param;
        offset_filler_param.set_max(2);
        offset_filler_param.set_min(0);
        UniformFiller<float> offset_filler(offset_filler_param);

        //offset_filler.Fill(&blob_offsets);
        const_zero_filer.Fill(&blob_offsets_x);
        const_zero_filer.Fill(&blob_offsets_y);

        const_zero_float_filer.Fill(&blob_offsets_float_x);
        const_zero_float_filer.Fill(&blob_offsets_float_y);

        const_zero_float_filer.Fill(&blob_output);

        const float* filtered_images = Caffe::mode() == Caffe::CPU ? blob_input.cpu_data() : blob_input.gpu_data();
        const float* error_images = Caffe::mode() == Caffe::CPU ? blob_error.cpu_data() : blob_error.gpu_data();
        const int* filter_offsets_x = Caffe::mode() == Caffe::CPU ? blob_offsets_x.cpu_data() : blob_offsets_x.gpu_data();
        const int* filter_offsets_y = Caffe::mode() == Caffe::CPU ? blob_offsets_y.cpu_data() : blob_offsets_y.gpu_data();
        const float* filter_offsets_float_x = Caffe::mode() == Caffe::CPU ? blob_offsets_float_x.cpu_data() : blob_offsets_float_x.gpu_data();
        const float* filter_offsets_float_y = Caffe::mode() == Caffe::CPU ? blob_offsets_float_y.cpu_data() : blob_offsets_float_y.gpu_data();
        const float* filter_weights = Caffe::mode() == Caffe::CPU ? blob_weights.cpu_data() : blob_weights.gpu_data();
        float* output = Caffe::mode() == Caffe::CPU ? blob_output.mutable_cpu_data() : blob_output.mutable_gpu_data();


        LayerParameter layer_param;
        FastAproxGaussianConvLayer<float> layer(layer_param);

        LayerParameter cudnn_layer_param;

        ConvolutionParameter* convolution_param =
                cudnn_layer_param.mutable_convolution_param();

        convolution_param->add_kernel_size(5);
        convolution_param->add_stride(1);
        convolution_param->add_pad(2);

        convolution_param->set_num_output(F);

        convolution_param->mutable_weight_filler()->set_type("gaussian");
        convolution_param->mutable_weight_filler()->set_std(0.01);

        std::cout << "input size:" << std::endl;
        std::cout << "blob_input: " << blob_input.count()  << std::endl;
        std::cout << "blob_offsets_x: " << blob_offsets_x.count()  << std::endl;
        std::cout << "blob_offsets_y: " << blob_offsets_y.count()  << std::endl;
        std::cout << "blob_weights: " << blob_weights.count()  << std::endl;
        std::cout << "blob_output: " << blob_output.count() << std::endl << std::endl;

        shared_ptr<CuDNNConvolutionLayer<float> > cudnn_layer(new CuDNNConvolutionLayer<float>(cudnn_layer_param));

        std::vector<Blob<float>* > blob_bottom_vec;
        std::vector<Blob<float>* > blob_top_vec;

        blob_bottom_vec.push_back(&blob_input);
        blob_top_vec.push_back(&blob_output);

        /*
        cudnn_layer->SetUp(blob_bottom_vec, blob_top_vec);

        clock_t start_t = clock();
        cudnn_layer->Forward(blob_bottom_vec, blob_top_vec);
        cudaDeviceSynchronize();
        clock_t end_t = clock();
        std::cout << "CuDNNConvolutionLayer forward pass in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl << std::endl;
        */
        for (int ii = 0; ii < 1; ++ii) {

            if (Caffe::mode() == Caffe::GPU)
                layer.test_backward_multi_subfeature_kernel_gpu(filtered_images, error_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, K, N, S, F, G, W, H, 5, 5, use_interpolation);

            float* output_c = blob_output.mutable_cpu_data();

            // verify data - since we use 1 for input and wights and 0 for offsets we should get N*W*H as output value for all

            int found_invalid = 0;

            //double WH = (double)W*(double)H;
            double WH = (double)W;

            //double GT_VALUE = (double)N*W*H;
            //double GT_VALUE = N*((WH)*((WH-1))/2); // input x error <== [1..N] x [1 1 .. 1]
            //double GT_VALUE = H*((WH)*((WH-1))/2); // input x error <== [1..N] x [1 1 .. 1]
            //double GT_VALUE = N*(((WH-1)*((WH-1)+1)*(2*(WH-1)+1))/6);
            double GT_VALUE = H*N*(((WH-1)*((WH-1)+1)*(2*(WH-1)+1))/6);

            for (int jj = 0; jj < blob_output.count(); ++jj) {
                if (output_c[jj] != GT_VALUE) {
                    if (found_invalid < 10)
                        printf("found invalid output (%f) at loc (%d) - should be %f\n", output_c[jj], jj, GT_VALUE);
                    found_invalid++;
                }
            }

            if (found_invalid > 0)
                printf("found num of invalid output vals: %d/%d\n",found_invalid, blob_output.count());
        }
    }

#ifdef USE_CUDNN

#endif

}  // namespace caffe
