#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

//#include "caffe/vision_layers.hpp"
#include "caffe/layers/gauss_conv_layer.hpp"
#include "caffe/util/math_functions_extra.hpp"

#include "caffe/util/fast_gauss_forward.hpp"
#include "caffe/util/fast_gauss_backward.hpp"


namespace caffe {

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
FastAproxGaussianConvLayer<Dtype>::~FastAproxGaussianConvLayer() {

}


template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::test_kernel_cpu(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
								const float* filter_weights, float* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height) {

	//caffe::fast_gauss_forward<float>(filtered_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
}


template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::test_kernel_gpu(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
								const float* filter_weights, float* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height) {

	caffe::fast_gauss_forward<float>(filtered_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
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
void FastAproxGaussianConvLayer<Dtype>::test_backward_multi_subfeature_kernel_gpu(const float* filtered_images, const float* error_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
																 const float* filter_weights, float* output,
																 const int K, const int I, const int S, const int F, const int G,
																 const int img_width, const int img_height,
																 const int kernel_width, const int kernel_height) {
	caffe::fast_gauss_backward_multi_subfeatures<float>(filtered_images, error_images, filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, I, S, F, G, K, img_width, img_height, kernel_width, kernel_height);
}


	template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_CLASS(FastAproxGaussianConvLayer);

}   // namespace caffe
#endif
