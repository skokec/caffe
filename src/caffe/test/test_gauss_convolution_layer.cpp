#include <boost/smart_ptr/shared_ptr.hpp>
#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/filler.hpp>
#include <caffe/layers/gauss_conv_layer.hpp>
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

  convolution_param->set_number_gauss(2);

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

  convolution_param->set_number_gauss(2);

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

  convolution_param->set_number_gauss(2);

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

  convolution_param->set_number_gauss(2);

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

  convolution_param->set_number_gauss(num_gauss);

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

/*
TYPED_TEST(GaussConvolutionLayerTest, Test0DConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  const int kNumOutput = 3;
  convolution_param->set_num_output(kNumOutput);
  convolution_param->set_axis(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  vector<int> top_shape = this->blob_bottom_->shape();
  top_shape[3] = kNumOutput;
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(top_shape, this->blob_top_->shape());
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  vector<int> weight_offset(2);
  const Blob<Dtype>* weight = layer->blobs()[0].get();
  const Blob<Dtype>* bias = layer->blobs()[1].get();
  const int num = this->blob_top_->count(3);
  const int dim = this->blob_top_->shape(3);
  const int bottom_dim = this->blob_bottom_->shape(3);
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < dim; ++d) {
      weight_offset[0] = d;
      Dtype value = bias->cpu_data()[d];
      for (int bottom_d = 0; bottom_d < bottom_dim; ++bottom_d) {
        weight_offset[1] = bottom_d;
        value += weight->data_at(weight_offset) *
                 this->blob_bottom_->cpu_data()[n * bottom_dim + bottom_d];
      }
      EXPECT_NEAR(value, this->blob_top_->cpu_data()[n * dim + d], 1e-4);
    }
  }
}

TYPED_TEST(GaussConvolutionLayerTest, TestSimple3DConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 5;
  bottom_shape[3] = this->blob_bottom_vec_[0]->shape(2);
  bottom_shape[4] = this->blob_bottom_vec_[0]->shape(3);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
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

TYPED_TEST(GaussConvolutionLayerTest, Test1x1Convolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
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
}

TYPED_TEST(GaussConvolutionLayerTest, TestSimpleConvolutionGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
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
}

TYPED_TEST(GaussConvolutionLayerTest, TestNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const int kernel_h = 11;
  const int kernel_w = 13;
  vector<int> bottom_shape(4);
  bottom_shape[0] = 15;
  bottom_shape[1] = 18;
  bottom_shape[2] = kernel_h * 2;
  bottom_shape[3] = kernel_w * 2;
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_num_output(12);
  convolution_param->set_bias_term(false);
  convolution_param->set_group(6);
  convolution_param->set_kernel_h(kernel_h);
  convolution_param->set_kernel_w(kernel_w);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  Blob<Dtype> weights;
  Blob<Dtype> top_diff;
  // Shape and fill weights and top_diff.
  bool copy_diff;
  bool reshape;
  {
    ConvolutionLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    top_diff.ReshapeLike(*this->blob_top_);
    filler.Fill(&top_diff);
    ASSERT_EQ(1, layer.blobs().size());
    copy_diff = false; reshape = true;
    weights.CopyFrom(*layer.blobs()[0], copy_diff, reshape);
  }
  vector<bool> propagate_down(1, true);
  Blob<Dtype> result_2d;
  Blob<Dtype> backward_result_2d;
  Blob<Dtype> backward_weight_result_2d;
  // Test with 2D im2col
  {
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
    caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result_2d.
    convolution_param->set_force_nd_im2col(false);
    ConvolutionLayer<Dtype> layer_2d(layer_param);
    layer_2d.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(1, layer_2d.blobs().size());
    copy_diff = false; reshape = false;
    layer_2d.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_2d.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false; reshape = true;
    result_2d.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    layer_2d.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_2d.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
  }
  Blob<Dtype> result_nd;
  Blob<Dtype> backward_result_nd;
  Blob<Dtype> backward_weight_result_nd;
  // Test with ND im2col
  {
    caffe_set(this->blob_top_->count(), Dtype(0),
              this->blob_top_->mutable_cpu_data());
    caffe_set(this->blob_bottom_->count(), Dtype(0),
              this->blob_bottom_->mutable_cpu_diff());
    caffe_set(weights.count(), Dtype(0), weights.mutable_cpu_diff());
    // Do SetUp and Forward; save Forward result in result_nd.
    convolution_param->set_force_nd_im2col(true);
    ConvolutionLayer<Dtype> layer_nd(layer_param);
    layer_nd.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(1, layer_nd.blobs().size());
    copy_diff = false; reshape = false;
    layer_nd.blobs()[0]->CopyFrom(weights, copy_diff, reshape);
    layer_nd.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    copy_diff = false; reshape = true;
    result_nd.CopyFrom(*this->blob_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_nd.
    ASSERT_EQ(this->blob_top_->shape(), top_diff.shape());
    caffe_copy(top_diff.count(), top_diff.cpu_data(),
               this->blob_top_->mutable_cpu_diff());
    layer_nd.Backward(this->blob_top_vec_, propagate_down,
                      this->blob_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_nd.CopyFrom(*this->blob_bottom_, copy_diff, reshape);
    backward_weight_result_nd.CopyFrom(weights, copy_diff, reshape);
  }
  ASSERT_EQ(result_nd.count(), result_2d.count());
  for (int i = 0; i < result_2d.count(); ++i)  {
    EXPECT_EQ(result_2d.cpu_data()[i], result_nd.cpu_data()[i]);
  }
  ASSERT_EQ(backward_result_nd.count(), backward_result_2d.count());
  for (int i = 0; i < backward_result_2d.count(); ++i) {
    EXPECT_EQ(backward_result_2d.cpu_diff()[i],
              backward_result_nd.cpu_diff()[i]);
  }
  ASSERT_EQ(backward_weight_result_nd.count(),
            backward_weight_result_2d.count());
  for (int i = 0; i < backward_weight_result_2d.count(); ++i) {
    EXPECT_EQ(backward_weight_result_2d.cpu_diff()[i],
              backward_weight_result_nd.cpu_diff()[i]);
  }
}

TYPED_TEST(GaussConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(GaussConvolutionLayerTest, TestGradient3D) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  vector<int> bottom_shape(5);
  bottom_shape[0] = this->blob_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->blob_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 5;
  bottom_shape[3] = this->blob_bottom_vec_[0]->shape(2);
  bottom_shape[4] = this->blob_bottom_vec_[0]->shape(3);
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  for (int i = 0; i < this->blob_bottom_vec_.size(); ++i) {
    this->blob_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->blob_bottom_vec_[i]);
  }
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(GaussConvolutionLayerTest, Test1x1Gradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(GaussConvolutionLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
*/
#ifdef USE_CUDNN

#endif

}  // namespace caffe
