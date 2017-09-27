#ifndef CAFFE_UTIL_FAST_GAUSS_BACKWARD_H_
#define CAFFE_UTIL_FAST_GAUSS_BACKWARD_H_


namespace caffe {
#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void fast_gauss_backward(const Dtype* filtered_images, const Dtype* error_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
						const Dtype* filter_weights, Dtype* output,
						const int I, const int S, const int F, const int G,
						const int img_width, const int img_height,
						const int kernel_width, const int kernel_height, cudaStream_t streamId = 0);

template <typename Dtype>
void fast_gauss_backward_multi_subfeatures(const Dtype* filtered_images, const Dtype* error_images,
										   const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y,
										   const Dtype* filter_weights,
										   Dtype* output,
										   const int I, const int S, const int F, const int G, const int K, const bool last_k_optional,
										   const int img_width, const int img_height,
										   const int kernel_width, const int kernel_height,
										   const bool use_interpolation, const bool ignore_edge_gradients,
										   Dtype* prepared_filtered_images, size_t* prepared_filtered_images_size,
										   Dtype* prepared_error_images, size_t* prepared_error_images_size,
										   Dtype* prepared_filter_weights, size_t* prepared_filter_weights_size,
                                           int* prepared_filter_offsets, size_t* prepared_filter_offsets_size, cudaStream_t streamId = 0);

class FastGaussBackwardMultiSubfeaturesCUDAParam {

public:
	// fixed params during construction
	const int img_width, img_height;
	const int I, S, F, G, K, IN_K;

	// parameters for setup before call

	// params for get_allocation_sizes call
	size_t* alloc_img, *alloc_err, *alloc_w, *alloc_off;

	// params for run_kernel call
	float const* filtered_images, *error_images, *filter_offsets_float_x, *filter_offsets_float_y, *filter_weights;
	float* output, *prepared_filtered_images, *prepared_error_images, *prepared_filter_weights;
	int* prepared_filter_offsets;
	int kernel_w, kernel_h;
	bool ignore_edge_gradients;
	cudaStream_t streamId;

public:
	FastGaussBackwardMultiSubfeaturesCUDAParam(const int img_width, const int img_height, const int I, const int S, const int F, const int G, const int K, const int IN_K) :
			img_width(img_width), img_height(img_height), I(I), S(S), F(F), G(G), K(K), IN_K(IN_K) {
	}
	void set_params_for_allocation_call(size_t* alloc_img, size_t* alloc_err, size_t* alloc_w, size_t* alloc_off) {
		this->alloc_img = alloc_img;
		this->alloc_w = alloc_w;
		this->alloc_err = alloc_err;
		this->alloc_off = alloc_off;
	}
	void set_params_for_kernel_call(const float* filtered_images, const float* error_images,
									const float* filter_offsets_float_x, const float* filter_offsets_float_y,
									const float* filter_weights, const int kernel_w, const int kernel_h,
									float* output,
									float* prepared_filtered_images,
									float* prepared_error_images,
									float* prepared_filter_weights,
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

};


// we make explicit functions for different combinations of
// each function is implemented in separate .cu file to allow for parallel compile
// (there are 288 combination all-together so this way we can reduce compute time by a factor of 8 if enough CPU cores)

void fast_gauss_backward_multi_subfeatures_patch_16x8(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_16x16(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_16x32(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_16x64(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_32x8(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_32x16(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_32x32(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_32x64(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_64x8(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_64x16(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_64x32(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);
void fast_gauss_backward_multi_subfeatures_patch_64x64(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackwardMultiSubfeaturesCUDAParam& PARAMS);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_FAST_GAUSS_BACKWARD_H_
