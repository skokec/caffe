#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <cmath>


#include "caffe/layers/fast_gauss/fast_gauss_forward.hpp"
#include "caffe/util/device_alternate.hpp"

#include "caffe/layers/fast_gauss/fast_gauss_forward_core.hpp"

namespace caffe {


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

template <>
void fast_gauss_forward<double>(const double* filtered_images,
								const double* filter_offsets_float_x, const double* filter_offsets_float_y, const double* filter_weights, int PARAM_FORMAT,
								double* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height, const bool use_interpolation,
								double* prepared_filtered_images, size_t* prepared_filtered_images_size,
								double* prepared_filter_weights, size_t* prepared_filter_weights_size,
								int* prepared_filter_offsets, size_t* prepared_filter_offsets_size,
								double* prepared_filter_offsets_and_weights, cudaStream_t streamId) {

}


template<>
void fast_gauss_forward<float>(const float* filtered_images,
//void fast_gauss_forward_new(const float* filtered_images,
							   const float* filter_offsets_float_x, const float* filter_offsets_float_y, const float* filter_weights, const int PARAM_FORMAT,
							   float* output,
							   const int I, const int S, const int F, const int G,
							   const int img_width, const int img_height,
							   const int kernel_width, const int kernel_height,const bool use_interpolation,
							   float* prepared_filtered_images, size_t* prepared_filtered_images_size,
							   float* prepared_filter_weights, size_t* prepared_filter_weights_size,
							   int* prepared_filter_offsets, size_t* prepared_filter_offsets_size,
							   float* prepared_filter_offsets_and_weights, cudaStream_t streamId) {

	FastGaussForwardCUDAParams params(img_width, img_height, I, S, F, G);

	params.set_params_for_allocation_call(prepared_filtered_images_size, prepared_filter_weights_size, prepared_filter_offsets_size);
	params.set_params_for_kernel_call(filtered_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, kernel_width, kernel_height, PARAM_FORMAT, output,
									  prepared_filtered_images, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights,
									  streamId);

	// calls either FastGaussForwardCUDA->run_kernel() or FastGaussForwardCUDA->get_allocation_sizes()
	// if prepared_filtered_images_size, prepared_filter_weights_size OR prepared_filter_offsets_size are not NULL

	// decide which size of patch to use to minimize wasted memory/processing
	int patch_size_w = img_width <= 16 ? 16 : select_optimal_block_size(img_width, 5, 6); // allowed patch sizes = 2^[5,6] i.e, [32,64]
	int patch_size_h = img_height <= 8 ? 8 :
						(img_height <= 16 ? 16 : select_optimal_block_size(img_height, 5, 6)); // allowed patch sizes = 2^[5,6] i.e, [32,64]


	// decide wheather to use:
	//  - 32 pixels per warp
	// 		- if 32x8 pixels and 1 images per block (full utilization)
	//  - 16 pixels per warp
	// 		- if 16x8 pixels and 2 images per block (full utilization)
	// 		- if 16x8 pixels and 1 images per block (half utilization)

	int boundry_img_width = img_width - floor(img_width/patch_size_w) * patch_size_w;

	int warp_pixel_size_x = min(patch_size_w, select_optimal_block_size(boundry_img_width, 4,5)); // allowed warp pixels sizes = 2^[4,5] ie, [16,32]

	int new_img_parts_width = (int)ceil((float)img_width / patch_size_w);
	int new_img_parts_height = (int)ceil((float)img_height / patch_size_h);

	int num_images = I * new_img_parts_width * new_img_parts_height;

	//int img_size_w = max(16, select_optimal_block_size(img_width, 3, 5)); // allowed sizes = 2^[3,4,5] i.e, 8,16,32
	//int img_size_h = max(8, select_optimal_block_size(img_height, 3, 5)); // allowed sizes = 2^[3,4,5] i.e, 8,16,32
	//int img_size_w = max(16, img_width >= 32 ? 32 : (img_width >= 16 ? 16 : 8 ));
	//int img_size_h = max(8, img_height >= 32 ? 32 : (img_height >= 16 ? 16 : 8 ));
	int max_offset = MAX(kernel_width, kernel_height)/2;

	// we compute multiple features by one thread but that depends on interpolation
	int batch_features = 8 * (use_interpolation ? 2 : 4);

	bool single_feature = F % batch_features == 0 ? false : true;
	bool single_subfeature = S % 2 == 0 ? false : true;

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

// EVALUATION/DEBUGGING code
//template <>
//void fast_gauss_forward<float>(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
void fast_gauss_forward_old(const float* filtered_images,
                            const float* filter_offsets_float_x, const float* filter_offsets_float_y,const float* filter_weights, const int PARAM_FORMAT,
                            float* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height,const bool use_interpolation,
							   float* prepared_filtered_images, size_t* prepared_filtered_images_size,
							   float* prepared_filter_weights, size_t* prepared_filter_weights_size,
							   int* prepared_filter_offsets, size_t* prepared_filter_offsets_size,
							   float* prepared_filter_offsets_and_weights, cudaStream_t streamId) {

    FastGaussForwardCUDAParams params(img_width, img_height, I, S, F, G);

    params.set_params_for_allocation_call(prepared_filtered_images_size, prepared_filter_weights_size, prepared_filter_offsets_size);
    params.set_params_for_kernel_call(filtered_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, kernel_width, kernel_height, PARAM_FORMAT, output,
                                      prepared_filtered_images, prepared_filter_weights, prepared_filter_offsets, prepared_filter_offsets_and_weights,
                                      streamId);
	if (1) {

		int img_size = MAX(img_width, img_height) >= 32 ? 32 : 16;
		int max_offset = MAX(kernel_width, kernel_height);

		// we will split image into patches of size [IMG_HEIGHT x IMG_WIDTH] so use that as image size, however,
		// we need to increase the number of images that will be process as each patch is now considered as one image
		// there is no need to recombine the output since we just sum over all patches to get gradients
		// NOTE:
		//	we make sure img size is not smaller then what a single block of cuda threads will use (i.e. 32x8)
		int new_img_parts_width = (int)ceil((float)img_width / max(32,img_size));
		int new_img_parts_height = (int)ceil((float)img_height / max(8,img_size));

		int num_images = I* new_img_parts_width * new_img_parts_height;

		FastGaussForwardCUDA<32,32, 8, 16, 2, true, false, false> _kernel_class(params);

		if (prepared_filtered_images_size != 0 || prepared_filter_weights_size != 0 || prepared_filter_offsets_size != 0) {
			_kernel_class.get_allocation_sizes(params);
			return;
		}

		float* prepared_filtered_images_;
		float* prepared_filter_weights_;
		int* prepared_filter_offsets_;
		float* prepared_filter_offsets_and_weights_;

		size_t prepared_filtered_images_size_,
				prepared_filter_weights_size_,
				prepared_filter_offsets_size_;


		_kernel_class.get_allocation_sizes(params);
		std::cout << "started malloc and memset" << std::endl;

		CUDA_CHECK(cudaMalloc(&prepared_filtered_images_, prepared_filtered_images_size_));
		CUDA_CHECK(cudaMemset(prepared_filtered_images_, 0,  prepared_filtered_images_size_));

		CUDA_CHECK(cudaMalloc(&prepared_filter_weights_, prepared_filter_weights_size_));

		CUDA_CHECK(cudaMalloc(&prepared_filter_offsets_, prepared_filter_offsets_size_));
		CUDA_CHECK(cudaMemset(prepared_filter_offsets_,0, prepared_filter_offsets_size_));

		CUDA_CHECK(cudaMalloc(&prepared_filter_offsets_and_weights_, prepared_filter_weights_size_+prepared_filter_offsets_size_));
		CUDA_CHECK(cudaMemset(prepared_filter_offsets_and_weights_,0, prepared_filter_weights_size_+prepared_filter_offsets_size_));



		std::cout << "waiting for mamlloc and memset" << std::endl;
		cudaDeviceSynchronize();

		std::cout << "started FastGaussForwardCUDA.run_kernel()" << std::endl;
		for (int i = 0; i < 1; ++i) {

			clock_t start_t = clock();
			_kernel_class.run_kernel(params);
			cudaDeviceSynchronize();
			clock_t end_t = clock();

			CUDA_POST_KERNEL_CHECK;
			std::cout << "FastGaussForwardCUDA.run_kernel() in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
		}
	}
	return;
    if (prepared_filtered_images_size != 0 || prepared_filter_weights_size != 0 || prepared_filter_offsets_size != 0) {
        *prepared_filtered_images_size = 1;
        *prepared_filter_weights_size = 1;
        *prepared_filter_offsets_size = 1;
		return;
    }


	static const int BATCH_PIXELS_SIZE_X = 1;
	static const int BATCH_PIXELS_SIZE_Y = 8;

	static const int PIXELS_INTERPOLATION_Dx = 2;
	static const int PIXELS_INTERPOLATION_Dy = 2;

	static const int BLOCK_X = 32/BATCH_PIXELS_SIZE_X;
	static const int BLOCK_Y = 8/BATCH_PIXELS_SIZE_Y;
	static const int BLOCK_FEATURES = 8;
    static const int BLOCK_IMAGES = 1;

	static const int BATCH_FEATURES_SIZE = 2;
	static const int BATCH_COMPUTE_FEATURES_SIZE = 2;

	static const int BATCH_COMPUTE_SUBFEATURES_SIZE = 1;
	static const int BATCH_MEM_SUBFEATURES_SIZE = 2;
	static const int BATCH_GAUSS_SIZE = 2;

	static const int BATCH_N = 1;

	static const int IMG_WIDTH = 32;
	static const int IMG_HEIGHT = 32;
	static const int MAX_OFFSET = 8;

	static const int NUM_SM = 1; // number of streaming multiprocessors

	typedef class BlockIndexing<NUM_SM,
					BLOCK_X, BLOCK_Y, BLOCK_FEATURES, BLOCK_IMAGES,
					BATCH_PIXELS_SIZE_X, BATCH_PIXELS_SIZE_Y,
					PIXELS_INTERPOLATION_Dx, PIXELS_INTERPOLATION_Dy,
					BATCH_FEATURES_SIZE,
					BATCH_COMPUTE_FEATURES_SIZE,
					BATCH_COMPUTE_SUBFEATURES_SIZE,
					BATCH_MEM_SUBFEATURES_SIZE,
					BATCH_GAUSS_SIZE,
					BATCH_N,
					IMG_WIDTH, IMG_HEIGHT,
					MAX_OFFSET,
			//false, 5, 2> BlockIndexingPipelineT;
			//false, 1, 6> BlockIndexingPipelineT;
			false, 4, 2> BlockIndexingPipelineT;

	int new_img_parts_width = (int)ceil((float)img_width / IMG_WIDTH);
	int new_img_parts_height = (int)ceil((float)img_height / IMG_HEIGHT);

	BlockIndexingPipelineT::Launch block_indexing;

	dim3 threadsPerBlock = block_indexing.getThreadsPerBlock(I * new_img_parts_width * new_img_parts_height, F, S, IMG_WIDTH, IMG_HEIGHT);

	dim3 numBlocks = block_indexing.getBlocksPerGrid(I * new_img_parts_width * new_img_parts_height, F, S, G, IMG_WIDTH, IMG_HEIGHT);

	float* filtered_images_with_border;

	FastForwardInputImage<BlockIndexingPipelineT> image_cuda_prepare(img_width, img_height, I, S, new_img_parts_width,new_img_parts_height);
	{
		CUDA_CHECK(cudaMalloc(&filtered_images_with_border, image_cuda_prepare.get_allocation_size()));
		CUDA_CHECK(cudaMemset(filtered_images_with_border, 0,  image_cuda_prepare.get_allocation_size()));
		cudaDeviceSynchronize();
		std::cout << "started create_input_with_border_bw" << std::endl;

		clock_t start_t = clock();
		filtered_images_with_border = image_cuda_prepare.create_input(filtered_images_with_border, filtered_images);
		cudaDeviceSynchronize();

		clock_t end_t = clock();
		CUDA_POST_KERNEL_CHECK;
		std::cout << "create_input_with_border_bw_multi in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

/*
		float* filtered_images_cpu = new float[(IMG_WIDTH + 2*MAX_OFFSET)*( IMG_HEIGHT + 2*MAX_OFFSET)* I*S * new_img_parts_width*new_img_parts_height];

		for (int i = 0; i < (IMG_WIDTH + 2*MAX_OFFSET)*( IMG_HEIGHT + 2*MAX_OFFSET)* I*S * new_img_parts_width*new_img_parts_height; ++i)
			filtered_images_cpu[i] = -1;

		cudaMemcpy(filtered_images_cpu, filtered_images_with_border, sizeof(float)* (IMG_WIDTH + 2*MAX_OFFSET)*( IMG_HEIGHT + 2*MAX_OFFSET)* I*S*new_img_parts_width*new_img_parts_height, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		//for (int i = 0; i < (img_width + 2*MAX_OFFSET)*( img_height + 2*MAX_OFFSET)* I*S; ++i) {
		int index =0;
		for (int i = 0; i < I/BATCH_N * new_img_parts_width*new_img_parts_height; ++i) {
			for (int s = 0; s < S; ++s) {
				for (int l =0; l < IMG_HEIGHT + 2*MAX_OFFSET; ++l){
					for (int n = 0; n < IMG_WIDTH + 2*MAX_OFFSET; ++n) {
						for (int i2 = 0; i2 < BATCH_N; ++i2)
							std::cout << filtered_images_cpu[index++] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << std:: endl << "end of s" << std::endl;
			}
			std::cout << std::endl;

		}
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << std::endl;
		//return;*/

		/*


		float* filtered_images_cpu = new float[(img_width)*( img_height )* I*S];

		for (int i = 0; i < img_width * img_height * I*S; ++i)
			filtered_images_cpu[i] = -1;

		cudaMemcpy(filtered_images_cpu, filtered_images, sizeof(int)* (img_width )*( img_height )* I*S, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		int index =0;
		for (int i = 0; i < I; ++i) {
			for (int s = 0; s < S; ++s) {
				for (int l =0; l < img_height; ++l){
					for (int n = 0; n < img_width; ++n)
						std::cout << filtered_images_cpu[index++] << " ";

					std::cout << std::endl;
				}
				std::cout << std:: endl << "end of s" << std::endl;
			}
			std::cout << std::endl;

		}
		std::cout << std::endl;*/
	}

	float* prepared_filter_weights_;
	int* prepared_filter_offsets_;

	float* prepared_filter_offsets_and_weights_;

	FastForwardInputWeightAndOffsets<BlockIndexingPipelineT> weight_and_offsets_cuda_prepare(img_width, img_height, I, F, S, G);

	{
		CUDA_CHECK(cudaMalloc(&prepared_filter_weights_, weight_and_offsets_cuda_prepare.get_weights_allocation_size()));

		CUDA_CHECK(cudaMalloc(&prepared_filter_offsets_, weight_and_offsets_cuda_prepare.get_offsets_allocation_size()));
		CUDA_CHECK(cudaMemset(prepared_filter_offsets_,0, weight_and_offsets_cuda_prepare.get_offsets_allocation_size()));

		CUDA_CHECK(cudaMalloc(&prepared_filter_offsets_and_weights_, weight_and_offsets_cuda_prepare.get_allocation_size()));
		CUDA_CHECK(cudaMemset(prepared_filter_offsets_and_weights_,0, weight_and_offsets_cuda_prepare.get_allocation_size()));

		cudaDeviceSynchronize();

		std::cout << "waiting for copy_permute_weights" << std::endl;

		clock_t start_t = clock();
		weight_and_offsets_cuda_prepare.create_input(prepared_filter_weights_, prepared_filter_offsets_, prepared_filter_offsets_and_weights_, filter_weights, filter_offsets_float_x, filter_offsets_float_y, kernel_width, kernel_height, PARAM_FORMAT);
		cudaDeviceSynchronize();

		clock_t end_t = clock();
		CUDA_POST_KERNEL_CHECK;
		std::cout << "copy_permute_weights in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

	}

	std::cout << "started fast_gauss_forward_pipeline_kernel" << std::endl;

	std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
	std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

//#define FIND_BEST_MEM_LOAD_DELAY
#ifndef FIND_BEST_MEM_LOAD_DELAY

	for (int i = 0; i < 30; ++i) {

		cudaMemset(output, 0, sizeof(float) * I * F * img_width * img_height);
		cudaDeviceSynchronize();

		clock_t start_t = clock();
		fast_gauss_forward_pipeline_kernel<BlockIndexingPipelineT,-1,-1><<<numBlocks,threadsPerBlock>>>(filtered_images_with_border, prepared_filter_offsets_, prepared_filter_weights_, prepared_filter_offsets_and_weights_, output, I, S, F, G, img_width, img_height, new_img_parts_width, new_img_parts_height);
		cudaDeviceSynchronize();

		clock_t end_t = clock();
		CUDA_POST_KERNEL_CHECK;
		std::cout << "fast_gauss_forward_pipeline_kernel in for "<< (((float)(end_t-start_t))/CLOCKS_PER_SEC)<< std::endl;
	}

#else

#define CALL_OP(TMP1, TMP2) \
	{ \
	float time_all = 0; \
	for (int i = 0; i < 50; ++i) { \
		clock_t start_t = clock(); \
		fast_gauss_forward_pipeline_kernel<BlockIndexingPipelineT,TMP1,TMP2><<<numBlocks,threadsPerBlock>>>(filtered_images_with_border, prepared_filter_offsets_, prepared_filter_weights_, prepared_filter_offsets_and_weights_, output, I, S, F, G, img_width, img_height, new_img_parts_width, new_img_parts_height); \
		cudaDeviceSynchronize(); \
		\
		clock_t end_t = clock(); \
		CUDA_POST_KERNEL_CHECK; \
		time_all += (((float)(end_t-start_t))/CLOCKS_PER_SEC); \
	}\
	std::cout << "fast_gauss_forward_pipeline_kernel in for tmp1: "<< TMP1 <<" and tmp2: "<< TMP2 <<" with time "<< time_all/50 << std::endl;\
	}
	CALL_OP(0,0);
	CALL_OP(0,1);
	CALL_OP(0,2);
	CALL_OP(0,3);
	CALL_OP(0,4);
	CALL_OP(0,5);
	CALL_OP(0,6);
	CALL_OP(0,7);


	CALL_OP(1,0);
	CALL_OP(1,1);
	CALL_OP(1,2);
	CALL_OP(1,3);
	CALL_OP(1,4);
	CALL_OP(1,5);
	CALL_OP(1,6);
	CALL_OP(1,7);

	CALL_OP(2,0);
	CALL_OP(2,1);
	CALL_OP(2,2);
	CALL_OP(2,3);
	CALL_OP(2,4);
	CALL_OP(2,5);
	CALL_OP(2,6);
	CALL_OP(2,7);

	CALL_OP(3,0);
	CALL_OP(3,1);
	CALL_OP(3,2);
	CALL_OP(3,3);
	CALL_OP(3,4);
	CALL_OP(3,5);
	CALL_OP(3,6);
	CALL_OP(3,7);

	CALL_OP(4,0);
	CALL_OP(4,1);
	CALL_OP(4,2);
	CALL_OP(4,3);
	CALL_OP(4,4);
	CALL_OP(4,5);
	CALL_OP(4,6);
	CALL_OP(4,7);

	CALL_OP(5,0);
	CALL_OP(5,1);
	CALL_OP(5,2);
	CALL_OP(5,3);
	CALL_OP(5,4);
	CALL_OP(5,5);
	CALL_OP(5,6);
	CALL_OP(5,7);

	CALL_OP(6,0);
	CALL_OP(6,1);
	CALL_OP(6,2);
	CALL_OP(6,3);
	CALL_OP(6,4);
	CALL_OP(6,5);
	CALL_OP(6,6);
	CALL_OP(6,7);
#endif
	//std::cout << "fast_gauss_forward_pipeline_kernel in " <<  << std::endl;

	cudaFree(prepared_filter_weights);
	cudaFree(prepared_filter_offsets);

	cudaFree(filtered_images_with_border);


}


}  // namespace caffe


