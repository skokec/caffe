#include <cmath>


#include "caffe/layers/fast_gauss/fast_gauss_forward.hpp"

namespace caffe {

#define MAX(x,y) (x > y ? x : y)

int select_optimal_block_size(int img_size, int min_power, int max_power) {
	float best_unutilized_percent = 1.0f;
	int best_block_size = 0;
	for (int i = min_power; i <= max_power; ++i) {
		int block_size = pow(2,i);

		float utilization_factor = (img_size / (float)block_size);
		float unutilized_percent = (ceil(utilization_factor) - utilization_factor);
		if (unutilized_percent <= best_unutilized_percent) {
			best_unutilized_percent = unutilized_percent;
			best_block_size = block_size;
		}
	}
	return best_block_size;
}

template <typename Dtype>
FastGaussForward<Dtype>::FastGaussForward(const int img_width, const int img_height, const int I, const int S, const int F, const int G, const bool use_interpolation)  :
        img_width(img_width), img_height(img_height), I(I), S(S), F(F), G(G), use_interpolation(use_interpolation) {

    // calls either FastGaussForwardCUDA->run_kernel() or FastGaussForwardCUDA->get_allocation_sizes()
    // if prepared_filtered_images_size, prepared_filter_weights_size OR prepared_filter_offsets_size are not NULL

    // decide which size of patch to use to minimize wasted memory/processing
    patch_size_w = img_width <= 16 ? 16 : select_optimal_block_size(img_width, 5, 6); // allowed patch sizes = 2^[5,6] i.e, [32,64]
    patch_size_h = img_height <= 8 ? 8 :
                       (img_height <= 16 ? 16 : select_optimal_block_size(img_height, 5, 6)); // allowed patch sizes = 2^[5,6] i.e, [32,64]


    // decide wheather to use:
    //  - 32 pixels per warp
    // 		- if 32x8 pixels and 1 images per block (full utilization)
    //  - 16 pixels per warp
    // 		- if 16x8 pixels and 2 images per block (full utilization)
    // 		- if 16x8 pixels and 1 images per block (half utilization)

    int boundry_img_width = img_width - floor(img_width/patch_size_w) * patch_size_w;

    warp_pixel_size_x = std::min(patch_size_w, select_optimal_block_size(boundry_img_width, 4,5)); // allowed warp pixels sizes = 2^[4,5] ie, [16,32]

    int new_img_parts_width = (int)ceil((float)img_width / patch_size_w);
    int new_img_parts_height = (int)ceil((float)img_height / patch_size_h);

    num_images = I * new_img_parts_width * new_img_parts_height;

    // we compute multiple features by one thread but that depends on interpolation
    int batch_features = 8 * (use_interpolation ? 2 : 4);

    single_feature = F % batch_features == 0 ? false : true;
    single_subfeature = S % 2 == 0 ? false : true;
}

template <typename Dtype>
void FastGaussForward<Dtype>::CUDAParams::set_params_for_allocation_call(size_t *alloc_img, size_t *alloc_w, size_t *alloc_off) {
    this->alloc_img = alloc_img;
    this->alloc_w = alloc_w;
    this->alloc_off = alloc_off;
}

template <typename Dtype>
void FastGaussForward<Dtype>::CUDAParams::set_params_for_kernel_call(const Dtype *filtered_images,
                                                                     const Dtype *filter_offsets_float_x, const Dtype *filter_offsets_float_y,
                                                                     const Dtype *filter_weights, const PARAM_FORMAT param_format, const int kernel_w, const int kernel_h,
                                                                     Dtype *output,
                                                                     Dtype *prepared_filtered_images,
                                                                     Dtype *prepared_filter_weights,
                                                                     int *prepared_filter_offsets,
                                                                     Dtype *prepared_filter_offsets_and_weights,
                                                                     cudaStream_t streamId) {
    this->filtered_images = filtered_images;
    this->filter_offsets_float_x = filter_offsets_float_x;
    this->filter_offsets_float_y = filter_offsets_float_y;
    this->filter_weights = filter_weights;
    this->kernel_w = kernel_w;
    this->kernel_h = kernel_h;
    this->param_format = param_format;
    this->output = output;
    this->prepared_filtered_images = prepared_filtered_images;
    this->prepared_filter_weights = prepared_filter_weights;
    this->prepared_filter_offsets = prepared_filter_offsets;
    this->prepared_filter_offsets_and_weights = prepared_filter_offsets_and_weights;
    this->streamId = streamId;
}
template <typename Dtype>
void FastGaussForward<Dtype>::get_allocation_sizes(const int kernel_width, const int kernel_height,
                                                   size_t* prepared_filtered_images_size,
                                                   size_t* prepared_filter_weights_size,
                                                   size_t* prepared_filter_offsets_size) {

    CUDAParams params(img_width, img_height, I, S, F, G);

    params.set_params_for_allocation_call(prepared_filtered_images_size, prepared_filter_weights_size, prepared_filter_offsets_size);

    params.set_params_for_kernel_call(NULL, NULL, NULL, NULL, PARAM_FORMAT::SGF, kernel_width, kernel_height, NULL,
                                      NULL, NULL, NULL, NULL, 0);

    call_cuda_kernel(params);
}


template <typename Dtype>
void FastGaussForward<Dtype>::forward_pass(const Dtype* filtered_images,
                                           const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
                                           const Dtype* filter_weights, const PARAM_FORMAT param_format, const int kernel_width, const int kernel_height,
                                           Dtype* output,
                                           Dtype* prepared_filtered_images,
                                           Dtype* prepared_filter_weights,
                                           int* prepared_filter_offsets,
                                           Dtype* prepared_filter_offsets_and_weights, cudaStream_t streamId) {

	CUDAParams params(img_width, img_height, I, S, F, G);

	params.set_params_for_allocation_call(NULL, NULL, NULL);
	params.set_params_for_kernel_call(filtered_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, param_format, kernel_width, kernel_height, output,
									  prepared_filtered_images, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights,
									  streamId);

    call_cuda_kernel(params);
}
template <>
void FastGaussForward<float>::call_cuda_kernel(CUDAParams& params) {

    int max_offset = MAX(params.kernel_w, params.kernel_h)/2;


	if (max_offset <= 4) {
		if (single_feature == false && single_subfeature == false) {
			// version where single_feature is false and single_subfeature false
			fast_gauss_forward_float_off_4_single_feat_0_single_subfeat_0(patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, params);

		} else if (single_feature == false && single_subfeature == true) {
			// version where single_feature is false and single_subfeature true
			fast_gauss_forward_float_off_4_single_feat_0_single_subfeat_1(patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, params);

		} else if (single_feature == true && single_subfeature == false) {
			// version where single_feature is true and single_subfeature false
			fast_gauss_forward_float_off_4_single_feat_1_single_subfeat_0(patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, params);

		} else {
			// version where single_feature is true and single_subfeature true
			fast_gauss_forward_float_off_4_single_feat_1_single_subfeat_1(patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, params);
		}
	} else if (max_offset <= 8) {
		if (single_feature == false && single_subfeature == false) {
			// version where single_feature is false and single_subfeature false
			fast_gauss_forward_float_off_8_single_feat_0_single_subfeat_0(patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, params);

		} else if (single_feature == false && single_subfeature == true) {
			// version where single_feature is false and single_subfeature true
			fast_gauss_forward_float_off_8_single_feat_0_single_subfeat_1(patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, params);

		} else if (single_feature == true && single_subfeature == false) {
			// version where single_feature is true and single_subfeature false
			fast_gauss_forward_float_off_8_single_feat_1_single_subfeat_0(patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, params);

		} else {
			// version where single_feature is true and single_subfeature true
			fast_gauss_forward_float_off_8_single_feat_1_single_subfeat_1(patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, params);
		}
	} else {
		printf("Unsupported filter size: %d. Supported only max up to 9x9 and 17x17 at the moment\n", max_offset);
        throw std::exception();
	}



	// CALL RUN_KERNEL_R4 macro that will call run_kernel() function on supplied class where first 4 parameters are replaced with compile-time known variables
	// replacing variables with compile-time known values allows CUDA compiler to generate kernels in advanced with pre-defined sizes
/*
	RUN_KERNEL_R7(FastGaussForwardCUDA, patch_size_w, patch_size_h, max_offset, warp_pixel_size_x, num_images, use_interpolation, single_feature, single_subfeature,
				  img_width, img_height, I, S, F, G,
				  filtered_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, kernel_width, kernel_height, PARAM_FORMAT, output,
				  prepared_filtered_images, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights,
				  streamId);
*/
}

template <>
void FastGaussForward<double>::call_cuda_kernel(CUDAParams& params) {
    throw std::exception();
}

template FastGaussForward<float>::FastGaussForward(const int img_width, const int img_height, const int I, const int S, const int F, const int G, const bool use_interpolation);
template FastGaussForward<double>::FastGaussForward(const int img_width, const int img_height, const int I, const int S, const int F, const int G, const bool use_interpolation);

template void FastGaussForward<float>::get_allocation_sizes(const int kernel_width, const int kernel_height, size_t* prepared_filtered_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);
template void FastGaussForward<double>::get_allocation_sizes(const int kernel_width, const int kernel_height, size_t* prepared_filtered_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);

template void FastGaussForward<float>::forward_pass(const float* filtered_images, const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, const PARAM_FORMAT param_format, const int kernel_width, const int kernel_height, float* output, float* prepared_filtered_images, float* prepared_filter_weights, int* prepared_filter_offsets, float* prepared_filter_offsets_and_weights, cudaStream_t streamId);
template void FastGaussForward<double>::forward_pass(const double* filtered_images, const double* filter_offsets_float_x, const double* filter_offsets_float_y, const double* filter_weights, const PARAM_FORMAT param_format, const int kernel_width, const int kernel_height, double* output, double* prepared_filtered_images, double* prepared_filter_weights, int* prepared_filter_offsets, double* prepared_filter_offsets_and_weights, cudaStream_t streamId);

}  // namespace caffe


