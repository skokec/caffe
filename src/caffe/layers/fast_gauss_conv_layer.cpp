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
																 const int kernel_width, const int kernel_height, const bool use_interpolation) {
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
														filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y,
														filter_weights, output,
														I, S, F, G, K,
														img_width, img_height,
														kernel_width, kernel_height,
														use_interpolation,
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

    cudaDeviceSynchronize();

	clock_t start_t = clock();

    caffe::fast_gauss_backward_multi_subfeatures<float>(filtered_images, error_images,
                                                        filter_offsets_x, filter_offsets_y, filter_offsets_float_x, filter_offsets_float_y,
                                                        filter_weights, output,
                                                        I, S, F, G, K,
                                                        img_width, img_height,
                                                        kernel_width, kernel_height,
														use_interpolation,
                                                        prepared_filtered_images,0,
                                                        prepared_error_images,0,
                                                        prepared_filter_weights,0,
                                                        prepared_filter_offsets,0);
	cudaDeviceSynchronize();
	clock_t end_t = clock();

	std::cout << "fast_gauss_backward_multi_subfeatures in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

	cudaFree(prepared_filter_weights);
	cudaFree(prepared_filter_offsets);

	cudaFree(prepared_filtered_images);
    cudaFree(prepared_error_images);

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
