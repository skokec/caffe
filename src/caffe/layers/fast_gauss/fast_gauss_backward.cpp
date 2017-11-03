#include <cmath>


#include "caffe/layers/fast_gauss/fast_gauss_backward.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {
/*

template <>
void fast_gauss_backward_multi_subfeatures<float>(const float* filtered_images, const float* error_images,
												  const float* filter_offsets_float_x, const float* filter_offsets_float_y,
												  const float* filter_weights,
												  float* output,
												  const int I, const int S, const int F, const int G, const int K, const bool last_k_optional,
												  const int img_width, const int img_height,
												  const int kernel_width, const int kernel_height,
                                                  const bool use_interpolation, const bool ignore_edge_gradients,
												  float* prepared_filtered_images, size_t* prepared_filtered_images_size,
												  float* prepared_error_images, size_t* prepared_error_images_size,
												  float* prepared_filter_weights, size_t* prepared_filter_weights_size,
												  int* prepared_filter_offsets, size_t* prepared_filter_offsets_size,
												  cudaStream_t streamId) {
*/
template <typename Dtype>
FastGaussBackward<Dtype>::FastGaussBackward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const bool last_k_optional, const bool use_interpolation) :
		img_width_in(img_width_in), img_height_in(img_height_in), img_width(img_width), img_height(img_height), I(I), S(S), F(F), G(G), IN_K(K), use_interpolation(use_interpolation) {

	// decide which size of patch to use to minimize wasted memory/processing
    if (img_width == 1 && img_height == 1) {
        patch_size_w = 1;
        patch_size_h = 1;
    } else {
	    patch_size_w = img_width <= 16 ? 16 : select_optimal_block_size_bw(img_width, 5, 6); // allowed patch sizes = 2^[5,6] i.e, [32,64]
	    patch_size_h = img_height <= 8 ? 8 :
				    	(img_height <= 16 ? 16 : select_optimal_block_size_bw(img_height, 5, 6)); // allowed patch sizes = 2^[5,6] i.e, [32,64]
	}

	// decide wheather to use:
	//  - 32 pixels per warp
	// 		- if 32x8 pixels and 1 images per block (full utilization)
	//  - 16 pixels per warp
	// 		- if 16x8 pixels and 2 images per block (full utilization)
	// 		- if 16x8 pixels and 1 images per block (half utilization)

	int boundary_img_width = img_width - floor(img_width/patch_size_w) * patch_size_w;

	int warp_pixel_size_x = patch_size_w == 1 ? 1 : std::min(patch_size_w, select_optimal_block_size_bw(boundary_img_width, 4,5)); // allowed warp pixels sizes = 2^[4,5] ie, [16,32]
	// NOTE:
	//	we make sure img size is not smaller then what a single block of cuda threads will use (i.e. 32x8)


	// we will split image into patches of size [IMG_HEIGHT x IMG_WIDTH] so use that as image size, however,
	// we need to increase the number of images that will be process as each patch is now considered as one image
	// there is no need to recombine the output since we just sum over all patches to get gradients

	int new_img_parts_width = (int)ceil((float)img_width / patch_size_w);
	int new_img_parts_height = (int)ceil((float)img_height / patch_size_h);

	num_images = I* new_img_parts_width * new_img_parts_height;

	single_subfeature = (S % 2 == 0 ? false : true);


	// last_k_optional==false and NUM_K==3
	// last_k_optional==true and NUM_K==4 and img_size_w >= 32 or
	//  - NUM_K = 3, BATCH_K_SIZE = 1, _WARP_PIXELS_X = 32
	//
	// last_k_optional==true and NUM_K==4 and img_size_w == 16 or
	//  - NUM_K = 4, BATCH_K_SIZE = 2, _WARP_PIXELS_X = 16

	use_smaller_warp_and_group_k = false;

	OUT_K = K;

	if (K == 4) {
		if (last_k_optional == false) {
			// we can automatically use 16 pixel warps and group K by 2
			use_smaller_warp_and_group_k = true;
		} else {
			// if last K is optional (i.e. we do not care for sigma) then decide to use 16 pixel warps only if our patch size is smaller
			use_smaller_warp_and_group_k = (warp_pixel_size_x < 32 ? true : false);
			// in case that we will be not be grouping then then we can skip last K since it appears to be optional
			// (NOTE: input K must remain the same to correctly load the data !!
			//        for output data we do not need to change anything since output has K as last dimension and we just ignore
			//        last K anyway)

			OUT_K = use_smaller_warp_and_group_k ? K : K - 1;
		}
	} else if (K == 3) {
		// if we have only K==3 then we cannot group K and instead use bigger warp size irrespectively of the patch size
		use_smaller_warp_and_group_k = false;
	} else {
		// not allowed
		printf("Unsupported K: %d. Supported only K=3 or K=4 at the moment\n", K);
		throw std::exception();
	}

	// if we do not use smaller warp then ensure patch_size_w is at least 32px
	if (use_smaller_warp_and_group_k == false)
		patch_size_w = std::max(32, patch_size_w);
}

template <typename Dtype>
int FastGaussBackward<Dtype>::select_optimal_block_size_bw(int img_size, int min_power, int max_power) {
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
void FastGaussBackward<Dtype>::CUDAParams::set_params_for_allocation_call(size_t* alloc_img, size_t* alloc_err, size_t* alloc_w, size_t* alloc_off) {
	this->alloc_img = alloc_img;
	this->alloc_w = alloc_w;
	this->alloc_err = alloc_err;
	this->alloc_off = alloc_off;
}

template <typename Dtype>
void FastGaussBackward<Dtype>::CUDAParams::set_params_for_kernel_call(const Dtype* filtered_images, const Dtype* error_images,
								const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
								const Dtype* filter_weights, const int kernel_w, const int kernel_h,
								Dtype* output,
								Dtype* prepared_filtered_images,
								Dtype* prepared_error_images,
								Dtype* prepared_filter_weights,
								int* prepared_filter_offsets,
								const bool ignore_edge_gradients,
								cudaStream_t streamId) {
	this->filtered_images = filtered_images;
	this->error_images = error_images;
	this->filter_offsets_float_x = filter_offsets_float_x;
	this->filter_offsets_float_y = filter_offsets_float_y;
	this->filter_weights = filter_weights;
	this->kernel_w = kernel_w;
	this->kernel_h = kernel_h;
	this->output = output;
	this->prepared_filtered_images = prepared_filtered_images;
	this->prepared_error_images = prepared_error_images;
	this->prepared_filter_weights = prepared_filter_weights;
	this->prepared_filter_offsets = prepared_filter_offsets;
	this->ignore_edge_gradients = ignore_edge_gradients;
	this->streamId = streamId;
}

template <typename Dtype>
void FastGaussBackward<Dtype>::get_allocation_sizes(const int kernel_width, const int kernel_height,
                                                                    size_t* prepared_filtered_images_size,
                                                                    size_t* prepared_error_images_size,
                                                                    size_t* prepared_filter_weights_size,
                                                                    size_t* prepared_filter_offsets_size) {

	CUDAParams params(img_width_in, img_height_in, img_width, img_height, I, S, F, G, OUT_K, IN_K);

	params.set_params_for_allocation_call(prepared_filtered_images_size, prepared_error_images_size, prepared_filter_weights_size, prepared_filter_offsets_size);
	params.set_params_for_kernel_call(NULL, NULL, NULL, NULL, NULL, kernel_width, kernel_height, NULL,
									  NULL, NULL, NULL, NULL, false, 0);

	call_cuda_kernel(params);
}

template <typename Dtype>
void FastGaussBackward<Dtype>::backward_pass(const Dtype* filtered_images, const Dtype* error_images,
													  const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
													  const Dtype* filter_weights,
													  const int kernel_width, const int kernel_height,
													  Dtype* output,
													  Dtype* prepared_filtered_images,
													  Dtype* prepared_error_images,
													  Dtype* prepared_filter_weights,
													  int* prepared_filter_offsets,
													  const bool ignore_edge_gradients,
													  cudaStream_t streamId) {

	CUDAParams params(img_width_in, img_height_in, img_width, img_height, I, S, F, G, OUT_K, IN_K);

	params.set_params_for_allocation_call(NULL, NULL, NULL, NULL);
	params.set_params_for_kernel_call(filtered_images, error_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, kernel_width, kernel_height, output,
									  prepared_filtered_images, prepared_error_images, prepared_filter_weights, prepared_filter_offsets, ignore_edge_gradients, streamId);

	call_cuda_kernel(params);
}
template <>
void FastGaussBackward<float>::call_cuda_kernel(CUDAParams& params) {


	int max_offset = MAX(params.kernel_w, params.kernel_h);

	// calls either FastGaussBackwardCUDA->run_kernel() or FastGaussBackwardCUDA->get_allocation_sizes()
	// if prepared_filtered_images_size, prepared_error_images_size, prepared_filter_weights_size OR prepared_filter_offsets_size are not NULL

	if (patch_size_h == 1 && patch_size_w == 1) {
        fast_gauss_backward_multi_subfeatures_patch_1x1(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
	} else if (patch_size_h >= 64) {
		if (patch_size_w >= 64) {
			fast_gauss_backward_multi_subfeatures_patch_64x64(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		} else if (patch_size_w >= 32) {
			fast_gauss_backward_multi_subfeatures_patch_32x64(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		} else {
			fast_gauss_backward_multi_subfeatures_patch_16x64(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		}
	} else if (patch_size_h >= 32) {
		if (patch_size_w >= 64) {
			fast_gauss_backward_multi_subfeatures_patch_64x32(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		} else if (patch_size_w >= 32) {
			fast_gauss_backward_multi_subfeatures_patch_32x32(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		} else {
			fast_gauss_backward_multi_subfeatures_patch_16x32(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		}
	} else if (patch_size_h >= 16) {
		if (patch_size_w >= 64) {
			fast_gauss_backward_multi_subfeatures_patch_64x16(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		} else if (patch_size_w >= 32) {
			fast_gauss_backward_multi_subfeatures_patch_32x16(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		} else {
			fast_gauss_backward_multi_subfeatures_patch_16x16(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		}
	} else {
		if (patch_size_w >= 64) {
			fast_gauss_backward_multi_subfeatures_patch_64x8(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		} else if (patch_size_w >= 32) {
			fast_gauss_backward_multi_subfeatures_patch_32x8(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		} else {
			fast_gauss_backward_multi_subfeatures_patch_16x8(patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params);
		}

	}



    // CALL RUN_KERNEL_R4 macro that will call run_kernel() function on supplied class where first 4 parameters are replaced with compile-time known variables
    // replacing variables with compile-time known values allows CUDA compiler to generate kernels in advanced with pre-defined sizes
	//RUN_KERNEL_R7(FastGaussBackwardCUDA, patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature, params)
/*
    RUN_KERNEL_R7(FastGaussBackwardCUDA, patch_size_w, patch_size_h, max_offset, use_smaller_warp_and_group_k, num_images, use_interpolation, single_subfeature,
                  img_width, img_height, I, S, F, G, OUT_K, K,
                  filtered_images, error_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, kernel_width, kernel_height, output,
                  prepared_filtered_images, prepared_error_images, prepared_filter_weights, prepared_filter_offsets, ignore_edge_gradients,
                  streamId)
*/

}

template <>
void FastGaussBackward<double>::call_cuda_kernel(CUDAParams& params) {

	throw std::exception();
}
/*
template <>
void fast_gauss_backward_multi_subfeatures<double>(const double* filtered_images, const double* error_images, const double* filter_offsets_float_x, const double* filter_offsets_float_y,
												   const double* filter_weights, double* output,
												   const int I, const int S, const int F, const int G, const int K, const bool is_last_k_optional,
												   const int img_width, const int img_height,
												   const int kernel_width, const int kernel_height,
                                                   const bool use_interpolation, const bool ignore_edge_gradients,
												   double* prepared_filtered_images, size_t* prepared_filtered_images_size,
												   double* prepared_error_images, size_t* prepared_error_images_size,
												   double* prepared_filter_weights, size_t* prepared_filter_weights_size,
                                                   int* prepared_filter_offsets, size_t* prepared_filter_offsets_size,
												   cudaStream_t streamId) {
}*/

template FastGaussBackward<float>::FastGaussBackward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const bool last_k_optional, const bool use_interpolation);
template FastGaussBackward<double>::FastGaussBackward(const int img_width_in, const int img_height_in, const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const bool last_k_optional, const bool use_interpolation);

template void FastGaussBackward<float>::get_allocation_sizes(const int kernel_width, const int kernel_height, size_t* prepared_filtered_images_size, size_t* prepared_error_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);
template void FastGaussBackward<float>::backward_pass(const float* filtered_images, const float* error_images, const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, const int kernel_width, const int kernel_height, float* output, float* prepared_filtered_images, float* prepared_error_images, float* prepared_filter_weights, int* prepared_filter_offsets, const bool ignore_edge_gradients, cudaStream_t streamId);

template void FastGaussBackward<double>::get_allocation_sizes(const int kernel_width, const int kernel_height, size_t* prepared_filtered_images_size, size_t* prepared_error_images_size, size_t* prepared_filter_weights_size, size_t* prepared_filter_offsets_size);
template void FastGaussBackward<double>::backward_pass(const double* filtered_images, const double* error_images, const double* filter_offsets_float_x, const double* filter_offsets_float_y, const double* filter_weights, const int kernel_width, const int kernel_height, double* output, double* prepared_filtered_images, double* prepared_error_images, double* prepared_filter_weights, int* prepared_filter_offsets, const bool ignore_edge_gradients, cudaStream_t streamId);
}  // namespace caffe
