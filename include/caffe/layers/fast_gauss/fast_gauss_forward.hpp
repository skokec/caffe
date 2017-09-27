#ifndef CAFFE_UTIL_FAST_GAUSS_FORWARD_H_
#define CAFFE_UTIL_FAST_GAUSS_FORWARD_H_


namespace caffe {
#ifndef CPU_ONLY  // GPU

enum { FAST_GAUSS_PARAM_SGF, FAST_GAUSS_PARAM_FGS};

template <typename Dtype>
void fast_gauss_forward(const Dtype* filtered_images,
                        const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y, const Dtype* filter_weights, int PARAM_FORMAT,
                        Dtype* output,
						const int I, const int S, const int F, const int G,
						const int img_width, const int img_height,
						const int kernel_width, const int kernel_height, const bool use_interpolation,
						Dtype* prepared_filtered_images, size_t* prepared_filtered_images_size,
						Dtype* prepared_filter_weights, size_t* prepared_filter_weights_size,
						int* prepared_filter_offsets, size_t* prepared_filter_offsets_size,
						Dtype* prepared_filter_offsets_and_weights, cudaStream_t streamId = 0);

class FastGaussForwardCUDAParams {

public:
	// fixed params during construction
	const int img_width, img_height;
	const int I, S, F, G;

	// parameters for setup before call

	// params for get_allocation_sizes call
	size_t* alloc_img, *alloc_w, *alloc_off;

	// params for run_kernel call
	float const* filtered_images, *filter_offsets_float_x, *filter_offsets_float_y, *filter_weights;
	float* output, *prepared_filtered_images, *prepared_filter_weights, *prepared_filter_offsets_and_weights;
	int* prepared_filter_offsets;
	int kernel_w, kernel_h;
	int PARAM_FORMAT;
	cudaStream_t streamId;

public:
	FastGaussForwardCUDAParams(const int img_width, const int img_height, const int I, const int S, const int F, const int G) :
			img_width(img_width), img_height(img_height), I(I), S(S), F(F), G(G) {
	}
	void set_params_for_allocation_call(size_t* alloc_img, size_t* alloc_w, size_t* alloc_off) {
		this->alloc_img = alloc_img;
		this->alloc_w = alloc_w;
		this->alloc_off = alloc_off;
	}
	void set_params_for_kernel_call(const float* filtered_images,
									const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights,
									const int kernel_w, const int kernel_h, const int PARAM_FORMAT,
									float* output,
									float* prepared_filtered_images,
									float* prepared_filter_weights,
									int* prepared_filter_offsets, float* prepared_filter_offsets_and_weights,
									cudaStream_t streamId) {
		this->filtered_images = filtered_images;
		this->filter_offsets_float_x = filter_offsets_float_x;
		this->filter_offsets_float_y = filter_offsets_float_y;
        this->filter_weights = filter_weights;
		this->kernel_w = kernel_w;
		this->kernel_h = kernel_h;
		this->PARAM_FORMAT = PARAM_FORMAT;
		this->output = output;
		this->prepared_filtered_images = prepared_filtered_images;
		this->prepared_filter_weights = prepared_filter_weights;
		this->prepared_filter_offsets = prepared_filter_offsets;
		this->prepared_filter_offsets_and_weights = prepared_filter_offsets_and_weights;
		this->streamId = streamId;
	}

};

// we make explicit functions for different combinations of [OFFSET, USE_SINGLE_FEATURE, USE_SINGLE_SUBFEATURE]
// each function is implemented in separate .cu file to allow for parallel compile
// (there are 288 combination all-together so this way we can reduce compute time by a factor of 8 if enough CPU cores)
void fast_gauss_forward_float_off_4_single_feat_0_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForwardCUDAParams& PARAMS);
void fast_gauss_forward_float_off_4_single_feat_0_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForwardCUDAParams& PARAMS);
void fast_gauss_forward_float_off_4_single_feat_1_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForwardCUDAParams& PARAMS);
void fast_gauss_forward_float_off_4_single_feat_1_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForwardCUDAParams& PARAMS);

void fast_gauss_forward_float_off_8_single_feat_0_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForwardCUDAParams& PARAMS);
void fast_gauss_forward_float_off_8_single_feat_0_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForwardCUDAParams& PARAMS);
void fast_gauss_forward_float_off_8_single_feat_1_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForwardCUDAParams& PARAMS);
void fast_gauss_forward_float_off_8_single_feat_1_single_subfeat_1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForwardCUDAParams& PARAMS);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_FAST_GAUSS_FORWARD_H_
