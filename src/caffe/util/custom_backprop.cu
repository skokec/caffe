#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <cmath>

#include "glog/logging.h"

#include "caffe/util/custom_backprop.hpp"
#include "caffe/util/custom_cub.cuh"
#include "caffe/util/device_alternate.hpp"

namespace caffe {


#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( l*num_k + k) * num_j + j)*num_i + i )

// CUSTOM convolution code for gradient computation
template <int Bx, int By, int Kx, int Ky, int NUM_FILTERS_PER_THREAD, int NUM_GAUSS_PER_THREAD, bool CHECK_PIXEL_BOUNDS>
__global__ void
//__launch_bounds__(128)
filterActs_YxX_color_kernel(const float* images, const float* error, const float* filters, float* output,
                                   const int I, const int S, const int F_, const int G_,
								   const int subfeat_i_, const int feat_i_, const int gauss_i_,
                                   const int img_width, const int img_height,
								   const int error_width, const int error_height,
								   const int kernel_width, const int kernel_height,
								   const int padding, const int stride) {

// INPUT: images  [I x S x H x W]
//		  error   [I x F x H x W]
//		  filters [S x G x F x Ky x Kx]
// OUTPUT output  [1 x S x G x F]

// TODO: using hardcoded warp size may not be portable (should use warpSize) but this way allows compiler optimization and avoids using dynamic memory allocation (for output_sum)
#define WARP_SIZE 32

//#define NUM_DERIVATIVE_FILTERS 4

//#define BATCH_PIXELS_SIZE 8 // good value (<70ms?)
#define BATCH_PIXELS_SIZE 4
//#define BATCH_PIXELS_SIZE 1

//#define BATCH_FILTERS_SIZE 4 	// good value (<70ms?)
//#define BATCH_FILTERS_SIZE 4	// (goes over G)
#define BATCH_FILTERS_STRUCT 4 	// 4 == float4, 2 == float2 (goes over F)

//#define NUM_FILTERS_PER_THREAD (BATCH_FILTERS_STRUCT*1) ////////////// TODO: change to 2 !!!!
//#define NUM_GAUSS_PER_THREAD BATCH_FILTERS_SIZE

#define NUM_WAPRS (Bx*By >= WARP_SIZE ? ((Bx*By) / WARP_SIZE) : 1)

#define PLOT_ENABLED (false)
#define SHOULD_PLOT(S) (S && PLOT_ENABLED && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0)

#ifndef CUBIN_EMBEDDING

	// Bx == size of block in x-dim (width)
	// By == size of block in y-dim (height)
	// Kx == size of kernel in x-dim (width)
	// Ky == size of kernel in y-dim (height)

	//const int subfeat_i = 0;
	//const int feat_i = 0;
	//const int gauss_i = 0;
	//const int img_i = 0;

	// calculate indexes for subfeat,feat and gauss
	// indexes are encoded in threadIdx.z
	const unsigned int F_GRID_DIM = (F_ + NUM_FILTERS_PER_THREAD -1) /  NUM_FILTERS_PER_THREAD;
	const unsigned int G_GRID_DIM = (G_ + NUM_GAUSS_PER_THREAD-1)    /  NUM_GAUSS_PER_THREAD;

	// feat_i MUST be multiple of 4 otherwise we cannot use float4 for copying output !!!
	const unsigned int subfeat_i = (blockIdx.z * blockDim.z + threadIdx.z) / (F_GRID_DIM * G_GRID_DIM);
	const unsigned int gauss_feat_i = (blockIdx.z * blockDim.z + threadIdx.z) % (F_GRID_DIM * G_GRID_DIM);
	const unsigned int feat_i = NUM_FILTERS_PER_THREAD *  (int)(gauss_feat_i / G_GRID_DIM);
	const unsigned int gauss_i = NUM_GAUSS_PER_THREAD * (gauss_feat_i % G_GRID_DIM);

// We load into shared memory original pixels within this block (multiplied by
// number of pixels per thread, i.e. BATCH_PIXELS_SIZE for width) plus added border
// for kernels. Borders are half of kernel on each size, where half is rounded down.
// Since we need borders on both side this is then 2*floor(Kx/2), effectivly Kx-1 for odd kernels (all kernels should be odd)
#define IMAGE_SH_PIXELS_X (Bx*BATCH_PIXELS_SIZE + Kx-1)
#define IMAGE_SH_PIXELS_Y (By + Ky-1)

#define IMAGE_SH_ARRAY_HEIGHT IMAGE_SH_PIXELS_Y

	// shared memory
#if BATCH_PIXELS_SIZE % 4 == 0

	// we can load 4 pixels with one LOAD operation so arrray needs to buffer be a multiple of 4
	// also make sure to add LOAD for last few pixels
	#define IMAGE_SH_ARRAY_WIDTH ((IMAGE_SH_PIXELS_X + 4-1)/4)
	// IMAGE_SH_PITCHED_WIDTH == actualy width of image_sh_align in floats
	#define IMAGE_SH_PITCHED_WIDTH (IMAGE_SH_ARRAY_WIDTH)*4

	__shared__ float4 image_sh_align[IMAGE_SH_ARRAY_HEIGHT][IMAGE_SH_ARRAY_WIDTH];

#elif BATCH_PIXELS_SIZE % 2 == 0

	#define IMAGE_SH_ARRAY_WIDTH ((IMAGE_SH_PIXELS_X + 2-1)/2)
	#define IMAGE_SH_PITCHED_WIDTH (IMAGE_SH_ARRAY_WIDTH)*2

	__shared__ float2 image_sh_align[IMAGE_SH_ARRAY_HEIGHT][IMAGE_SH_ARRAY_WIDTH];

#else

	#define IMAGE_SH_ARRAY_WIDTH ((IMAGE_SH_PIXELS_X + 1-1)/1)
	#define IMAGE_SH_PITCHED_WIDTH (IMAGE_SH_ARRAY_WIDTH)*1

	__shared__ float image_sh_align[IMAGE_SH_ARRAY_HEIGHT][IMAGE_SH_ARRAY_WIDTH];

#endif

	float* image_sh = reinterpret_cast<float*>(image_sh_align);

	__shared__ float4 kernel_sh[Ky][Kx][NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT][NUM_GAUSS_PER_THREAD];
	__shared__ float4 output_sum[NUM_WAPRS][NUM_GAUSS_PER_THREAD][NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT];
	__shared__ unsigned int image_sh_kernel_offsets[Ky*Kx+2];


	int offset_x = BATCH_PIXELS_SIZE * (blockIdx.x * Bx);
	int offset_y = blockIdx.y * By;
	//int offset_x = BATCH_PIXELS_SIZE * (blockIdx.x * Bx + threadIdx.x);
	//int offset_y = blockIdx.y * By + threadIdx.y;

	// load shared memory

	// let first thread load all kernel values
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {


		for (unsigned int j = 0; j < Ky; ++j) {
			for (unsigned int i = 0; i < Kx; ++i) {
				// offset values for each kernel from the start of (0,0) in kernel applied to image_sh buffer
				// i.e. it encodes how many steps to move in image_sh buffer to get to the pixel neccessary for multiplication with kernel at (i,j)
				image_sh_kernel_offsets[j*Kx + i] = i + j*(IMAGE_SH_PITCHED_WIDTH);
			}
		}

		for (unsigned int w = 0; w < NUM_WAPRS; ++w) {
			for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
				for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT; ++f) {
					output_sum[w][g][f].x = 0; output_sum[w][g][f].y = 0; output_sum[w][g][f].z = 0; output_sum[w][g][f].w = 0;
				}
			}
		}
	}
	// Specialize WarpReduce for type int
	typedef cub::WarpReduce<float> WarpReduce;

	// Allocate WarpReduce shared memory for one warp
	__shared__ typename WarpReduce::TempStorage warp_reduce_storage[NUM_WAPRS];

	int warp_id = threadIdx.x * threadIdx.y * threadIdx.z / warpSize;
	WarpReduce warp_reduce(warp_reduce_storage[warp_id]);


	// point to specific output and filter
	float* current_filter = (float*)filters + OFFSET(subfeat_i,gauss_i,feat_i,0, S,G_,F_,Kx*Ky); //( ( (subfeat_i)*G + gauss_i)*F + feat_i)*(Kx*Ky);

	// load kernel
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		for (unsigned int j = 0; j < Ky; ++j) {
			for (unsigned int i = 0; i < Kx; ++i) {
				for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
					for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD/BATCH_FILTERS_STRUCT; ++f) {
						float4 kernel_tmp;
						kernel_tmp.x = current_filter[ (( g*F_ + f*BATCH_FILTERS_STRUCT + 0)*Ky + j)*Kx + i];
						kernel_tmp.y = current_filter[ (( g*F_ + f*BATCH_FILTERS_STRUCT + 1)*Ky + j)*Kx + i];
						kernel_tmp.z = current_filter[ (( g*F_ + f*BATCH_FILTERS_STRUCT + 2)*Ky + j)*Kx + i];
						kernel_tmp.w = current_filter[ (( g*F_ + f*BATCH_FILTERS_STRUCT + 3)*Ky + j)*Kx + i];
						kernel_sh[j][i][f][g] = kernel_tmp;
					}
				}
			}
		}
	}

	if (SHOULD_PLOT(1)) {
		printf("all kernels:\n");
		for (int i = 0; i < Ky*Kx*G_*F_; ++i) {
			printf("%f, ",current_filter[i]);
			if ((i % (Ky*Kx)) == Kx*Ky-1) printf("\n\n");
		}

	}
	if (PLOT_ENABLED) {
		printf("threadIdx.z:%d, subfeat_i: %d/%d, gauss_i: %d/%d, feat_i: %d/%d\n",blockIdx.z * blockDim.z + threadIdx.z,subfeat_i,S, gauss_i, G_, feat_i, F_);
	}

	// go over all images
	for (unsigned int img_i = 0; img_i < I; ++img_i) {

		// point to specific image
		float* current_image = (float*)images + OFFSET(img_i,subfeat_i,0,0, I,S,img_height,img_width); //( img_i*S + subfeat_i)*(img_width * img_height);

		// move image and error pointer to position for this block
		current_image += (img_width * (offset_y) + (offset_x));

		// load image with apron (poor utilization when big kernels)

		int start_j = threadIdx.y - Ky/2;
		int end_j = By + Ky/2;
		int start_i = threadIdx.x*BATCH_PIXELS_SIZE - Kx/2;
		int end_i = Bx*BATCH_PIXELS_SIZE + Kx/2;

		#pragma unroll
		for (int j = start_j; j < end_j; j+=By) {
			#pragma unroll
			for (int i = start_i; i < end_i; i+=Bx*BATCH_PIXELS_SIZE) {
				// current_image already at position for this block
				#define IS_VALID_PIXEL(X,Y,MAX_X,MAX_Y) (X >= 0 && X < MAX_X && Y >= 0 && Y < MAX_Y)

				if (BATCH_PIXELS_SIZE % 4 == 0) {
					#pragma unroll
					for (int k = 0; k < BATCH_PIXELS_SIZE/4; ++k) {
						float4 tmp;
						tmp.x = IS_VALID_PIXEL(i + 0 + k*4 + offset_x, j + offset_y, img_width, img_height) ? current_image[j * img_width + i + k*4 + 0] : 0;
						tmp.y = IS_VALID_PIXEL(i + 1 + k*4 + offset_x, j + offset_y, img_width, img_height) ? current_image[j * img_width + i + k*4 + 1] : 0;
						tmp.z = IS_VALID_PIXEL(i + 2 + k*4 + offset_x, j + offset_y, img_width, img_height) ? current_image[j * img_width + i + k*4 + 2] : 0;
						tmp.w = IS_VALID_PIXEL(i + 3 + k*4 + offset_x, j + offset_y, img_width, img_height) ? current_image[j * img_width + i + k*4 + 3] : 0;
						reinterpret_cast<float4*>(image_sh)[(j + Ky/2)*IMAGE_SH_ARRAY_WIDTH + (i + Kx/2 + k)/4] = tmp;
					}
				} else if (BATCH_PIXELS_SIZE % 2 == 0) {
					for (int k = 0; k < BATCH_PIXELS_SIZE/2; ++k) {
						float2 tmp;
						tmp.x = IS_VALID_PIXEL(i + 0 + k*2 + offset_x, j + offset_y, img_width, img_height) ? current_image[j * img_width + i + k*2 + 0] : 0;
						tmp.y = IS_VALID_PIXEL(i + 1 + k*2 + offset_x, j + offset_y, img_width, img_height) ? current_image[j * img_width + i + k*2 + 1] : 0;
						reinterpret_cast<float2*>(image_sh)[(j + Ky/2)*IMAGE_SH_ARRAY_WIDTH + (i + Kx/2 + k)/2] = tmp;
					}
				} else {
					for (int k = 0; k < BATCH_PIXELS_SIZE; ++k) {
						image_sh[(j + Ky/2)*IMAGE_SH_ARRAY_WIDTH + (i + Kx/2 + k)] = IS_VALID_PIXEL(i + k + offset_x, j + offset_y, img_width, img_height) ? current_image[j * img_width + i + k] : 0;
					}
				}
			}
		}

		// make sure all threads have finished loading memory
		__syncthreads();


		// thread participates only if its pixels are within valid bounds
		int pixel_loc_x = BATCH_PIXELS_SIZE * (blockIdx.x * Bx + threadIdx.x);
		int pixel_loc_y = blockIdx.y * By + threadIdx.y;

		if (pixel_loc_x < error_width && pixel_loc_y < error_height) {


			if (SHOULD_PLOT(1)) {
				printf("image ptr: %p, current ptr: %p\n",images, current_image);
				printf("img_i: %d, subfeat_i: %d \n",img_i, subfeat_i);
				printf("offset_x: %d, offset_y: %d \n",offset_x, offset_y);
				printf("image: %d/%d \n",img_i, I);
				printf("I %d ,S %d ,img_height %d, img_width %d\n", I,S,img_height,img_width);
				for (int j = 0; j < img_height; ++j) {
					for (int i = 0; i < img_width; ++i) {
						printf("%f, ",images[j*img_width + i]);
					}
					printf("\n");
				}
				printf("current_image: %d/%d \n",img_i, I);
				for (int j = 0; j < img_height; ++j) {
					for (int i = 0; i < img_width; ++i) {
						printf("%f, ",current_image[j*img_width + i]);
					}
					printf("\n");
				}
				printf("shared image: %d/%d \n",img_i, I);
				for (int j = 0; j < IMAGE_SH_PIXELS_Y; ++j) {
					for (int i = 0; i < IMAGE_SH_ARRAY_WIDTH*4; ++i) {
						printf("%f, ",image_sh[j*IMAGE_SH_ARRAY_WIDTH*4  +  i]);
					}
					printf("\n");
				}
				printf("shared image kernel offsets: \n");
				for (int j = 0; j < kernel_height; ++j) {
					for (int i = 0; i < kernel_width; ++i) {
						printf("%d, ", (int)image_sh_kernel_offsets[j*Kx  +  i]);
					}
					printf("\n");
				}

			}

#define BATCH_DOT 2
#define BATCH_DOT_STORE 1
			// prepare pointers for image data - hold pointers for two loops (where loop goes over kernel sizes)
			float* img_sh_p = &image_sh[(threadIdx.y)*IMAGE_SH_PITCHED_WIDTH + BATCH_PIXELS_SIZE*threadIdx.x];
			float* img_sh_p_tmp[BATCH_DOT];


			// load pointers to N consecetive pixels in image_sh where N=BATCH_DOT, should be N=2
			// this enables double buffering i.e. process data from one loop and load data into registers for next one
			#pragma unroll
			for (int k = 0; k < BATCH_DOT; ++k) {
				img_sh_p_tmp[k] = img_sh_p + image_sh_kernel_offsets[k];
			}

			float* current_error = (float*)error + OFFSET(img_i,feat_i,offset_y+threadIdx.y,offset_x+threadIdx.x*BATCH_PIXELS_SIZE, I,F_,error_height,error_width); //( img_i*F + feat_i)*(error_width * error_height);

			if (SHOULD_PLOT(1)) {
				for (int ff = 0; ff < F_; ++ff) {
				float* plot_error = (float*)error + OFFSET(img_i,ff,0,0, I,F_,error_height,error_width);
					printf("error_image: %d/%d \n",ff, F_);
					for (int j = 0; j < error_height; ++j) {
						for (int i = 0; i < error_width; ++i) {
							printf("%f, ",plot_error[j*img_width + i]);
						}
						printf("\n");
					}
				}
			}

			#pragma unroll
			for (int f = 0; f < NUM_FILTERS_PER_THREAD/BATCH_FILTERS_STRUCT; ++f) {

				// prepare pointers for kernel/filters data - hold pointers for two loops (where loop goes over kernel sizes)
				// get pointer to start of kernels
				float4* kr_sh_p = &kernel_sh[0][0][f][0];// [Kx][Ky][NUM_FILTERS_PER_THREAD/BATCH_FILTERS_STRUCT][NUM_GAUSS_PER_THREAD].[BATCH_FILTERS_STRUCT]
				float4* kr_sh_p_tmp[BATCH_DOT];

				#pragma unroll
				for (int k = 0; k < BATCH_DOT; ++k) {
					kr_sh_p_tmp[k] = kr_sh_p + NUM_FILTERS_PER_THREAD * NUM_GAUSS_PER_THREAD / BATCH_FILTERS_STRUCT * k;
				}

				// define intermediate values - they should be loaded to registers to allow for fast access
				// since too high num of registers is limted (max 255) and reduces occupancy we cannot to afford to many
				// note, NUM_GAUSS_PER_THREAD == NUM_GAUSS_PER_THREAD while BATCH_FILTERS_STRUCT goes over F
				float4 intermediate_sum[BATCH_PIXELS_SIZE][NUM_GAUSS_PER_THREAD];
				float4 loaded_kernel_vals[BATCH_DOT][NUM_GAUSS_PER_THREAD];

				// make sure to start with zeros
				#pragma unroll
				for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
					#pragma unroll
					for (int i = 0; i < NUM_GAUSS_PER_THREAD; ++i) {
						intermediate_sum[jj][i].x = 0; intermediate_sum[jj][i].y = 0; intermediate_sum[jj][i].z = 0; intermediate_sum[jj][i].w = 0;
					}
				}

				// then pre-start the main loop by loading elements for the first loop
				float img_tmp[BATCH_DOT][BATCH_PIXELS_SIZE];
				{
					int data_loading_k = 0;

					#pragma unroll
					for (int ii = 0; ii < BATCH_PIXELS_SIZE; ++ii) {
						if (SHOULD_PLOT(1)) printf("access offset: %d\n", (int)(img_sh_p_tmp[data_loading_k]+ii -  (float*)image_sh));

						img_tmp[data_loading_k][ii] = img_sh_p_tmp[data_loading_k][ii];
					}

					#pragma unroll
					for (int ii = 0; ii < NUM_GAUSS_PER_THREAD; ++ii) {
						loaded_kernel_vals[data_loading_k][ii] = kr_sh_p_tmp[data_loading_k][ii];
					}
				}

				// main loop that goes over kernel and image values (does dot product)
				// should only go to last-1 and ignore last one since Kx*Ky will always be odd (assuming BATCH_DOT=2)
				// we then need to handle odd last element seperately
				for (int ij = 0; ij < Ky*Kx - BATCH_DOT; ij+=BATCH_DOT) {

					if (SHOULD_PLOT(1)) {
						printf("ij=%d\n", ij);
					}

					#pragma unroll
					for (int k = 0; k < BATCH_DOT; ++k) {
						const int processing_k = k;
						const int data_loading_k = (k + 1) % BATCH_DOT;
						const int pointer_loading_k = (k + 2) % BATCH_DOT;

						// move loading pointers for loading in next loop
						img_sh_p_tmp[pointer_loading_k] = img_sh_p + image_sh_kernel_offsets[ij+ k + 2];
						kr_sh_p_tmp[pointer_loading_k] = kr_sh_p_tmp[data_loading_k] + NUM_FILTERS_PER_THREAD * NUM_GAUSS_PER_THREAD / BATCH_FILTERS_STRUCT;

						// start loading elements for next loop
						{
							for (int ii = 0; ii < BATCH_PIXELS_SIZE; ++ii) {
								if (SHOULD_PLOT(1)) printf("access offset: %d\n", (int)(img_sh_p_tmp[data_loading_k]+ii -  (float*)image_sh));

								img_tmp[data_loading_k][ii] = img_sh_p_tmp[data_loading_k][ii];
							}

							#pragma unroll
							for (int ii = 0; ii < NUM_GAUSS_PER_THREAD; ++ii) {
								loaded_kernel_vals[data_loading_k][ii] = kr_sh_p_tmp[data_loading_k][ii];
							}

						}
						// start processing element(s) for current loop
						#pragma unroll
						for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
							#pragma unroll
							for (int ii = 0; ii < NUM_GAUSS_PER_THREAD; ++ii) {
								if (SHOULD_PLOT(1)) printf("%f x %f, ", img_tmp[processing_k][jj], loaded_kernel_vals[processing_k][ii].x);

								intermediate_sum[jj][ii].x +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].x;
								intermediate_sum[jj][ii].y +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].y;
								intermediate_sum[jj][ii].z +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].z;
								intermediate_sum[jj][ii].w +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].w;
							}
						}

						if (SHOULD_PLOT(1)) {
							printf("\n");
						}

					}
				}

				// load pointers to N consecetive pixels in image_sh where N=BATCH_DOT, should be N=2
				// this enables double buffering i.e. process data from one loop and load data into registers for next one
				#pragma unroll
				for (int k = 0; k < BATCH_DOT; ++k) {
					img_sh_p_tmp[k] = img_sh_p + image_sh_kernel_offsets[k];
				}

				// load error values a bit sooner then they are needed to hide global memory access latency (usually ~200 cycles in maxwell)
				float error_values[BATCH_FILTERS_STRUCT][BATCH_PIXELS_SIZE];

				const uint error_img_size = error_height*error_width;

				#pragma unroll
				for (int kk = 0; kk < BATCH_FILTERS_STRUCT; ++kk) {
					#pragma unroll
					for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
						// also make sure to use 0 if this pixel is out of bounds
						error_values[jj][kk] = CHECK_PIXEL_BOUNDS == 0 || pixel_loc_x + jj < error_width ? (current_error[jj]) : 0;
					}
					current_error += error_img_size;
				}

				// process last element as well (should be already loaded in k=0)
				{
					int processing_k = 0;

					#pragma unroll
					for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
						#pragma unroll
						for (int ii = 0; ii < NUM_GAUSS_PER_THREAD; ++ii) {
							if (SHOULD_PLOT(1)) printf("%f x %f, ", img_tmp[processing_k][jj], loaded_kernel_vals[processing_k][ii].x);

							intermediate_sum[jj][ii].x +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].x;
							intermediate_sum[jj][ii].y +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].y;
							intermediate_sum[jj][ii].z +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].z;
							intermediate_sum[jj][ii].w +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].w;
						}
					}
				}

				if (SHOULD_PLOT(1)){
					for (int ii = 0; ii < NUM_GAUSS_PER_THREAD; ++ii) {
						for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
							printf("%f, ", intermediate_sum[jj][ii].x);
							printf("%f, ", intermediate_sum[jj][ii].y);
							printf("%f, ", intermediate_sum[jj][ii].z);
							printf("%f, ", intermediate_sum[jj][ii].w);
							printf("\n");
						}
					}
				}
				if (SHOULD_PLOT(1)) {
					printf("error mul:\n");
				}

				// sum over current warp to avoid concurrent memory writes
				// we can multiply with errors all intermediate_sums
				float4 filter_sums[NUM_GAUSS_PER_THREAD];
				#pragma unroll
				for (int ii = 0; ii < NUM_GAUSS_PER_THREAD; ++ii) {
					filter_sums[ii].x = 0;
					filter_sums[ii].y = 0;
					filter_sums[ii].z = 0;
					filter_sums[ii].w = 0;
					#pragma unroll
					for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
						filter_sums[ii].x += intermediate_sum[jj][ii].x * error_values[jj][0];
						filter_sums[ii].y += intermediate_sum[jj][ii].y * error_values[jj][1];
						filter_sums[ii].z += intermediate_sum[jj][ii].z * error_values[jj][2];
						filter_sums[ii].w += intermediate_sum[jj][ii].w * error_values[jj][3];

						if (SHOULD_PLOT(1)) {
							printf("%f x %f, ", intermediate_sum[jj][ii].x , error_values[jj][0]);
							printf("%f x %f, ", intermediate_sum[jj][ii].y , error_values[jj][1]);
							printf("%f x %f, ", intermediate_sum[jj][ii].z , error_values[jj][2]);
							printf("%f x %f, ", intermediate_sum[jj][ii].w , error_values[jj][3]);
						}

					}

					if (PLOT_ENABLED){
						printf("intermedite sum sum: %f, \n", filter_sums[ii].x);
					}
					if (SHOULD_PLOT(1)) {
						printf("sum values: %f, sum_out: %f\n", filter_sums[ii].x, warp_reduce.Sum(filter_sums[ii].x));
						printf("\n");
					}

					filter_sums[ii].x = warp_reduce.Sum(filter_sums[ii].x);
					filter_sums[ii].y = warp_reduce.Sum(filter_sums[ii].y);
					filter_sums[ii].z = warp_reduce.Sum(filter_sums[ii].z);
					filter_sums[ii].w = warp_reduce.Sum(filter_sums[ii].w);
				}

				if (SHOULD_PLOT(1)) {
					printf("filter sum:\n");
					for (int ii = 0; ii < NUM_GAUSS_PER_THREAD; ++ii) {
						printf("%f, ",filter_sums[ii].x);
						printf("%f, ",filter_sums[ii].y);
						printf("%f, ",filter_sums[ii].z);
						printf("%f, ",filter_sums[ii].w);
						printf("\n");
					}
				}

				const int tid = threadIdx.x +
						  threadIdx.y * Bx +
						  threadIdx.z * Bx * By;// +
						  //blockIdx.x * Bx * By * blockDim.z;

				if (tid  % WARP_SIZE == 0) {
					const int warp_id = tid/ WARP_SIZE;

					if (SHOULD_PLOT(1)) {
						printf("writing to output_sum with warp_id: %d\n", warp_id);
					}

					#pragma unroll
					for (int i = 0; i < NUM_GAUSS_PER_THREAD; ++i) {
						// note, NUM_GAUSS_PER_THREAD == NUM_GAUSS_PER_THREAD
						float4 tmp = output_sum[warp_id][i][f];

						tmp.x += filter_sums[i].x;
						tmp.y += filter_sums[i].y;
						tmp.z += filter_sums[i].z;
						tmp.w += filter_sums[i].w;

						if (SHOULD_PLOT(1)) {
							printf("%f, ",tmp.x);
							printf("%f, ",tmp.y);
							printf("%f, ",tmp.z);
							printf("%f, ",tmp.w);
							printf("\n");
						}

						output_sum[warp_id][i][f] = tmp;
					}

				}
			}
		}

		// make sure all threads have finished computation before loading next image
		// TODO: we could use double buffering to preload next image and avoid sync, but this would require more shared memory per block
		__syncthreads();
	}

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		// NOTE: must be synced from previous loop !!

		// F and feat_i MUST be a multiple of 4 !!!
		float4* current_output = reinterpret_cast<float4*>((float*)output + OFFSET(0,subfeat_i,gauss_i,feat_i, 1,S,G_,F_)); //( (  subfeat_i * G + gauss_i)*F + feat_i )

		// let first thread sum over results of all warps for each output i.e., (G,F) pair
		float4 final_output_sum[NUM_GAUSS_PER_THREAD][NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT];

		for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
			for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT; ++f) {
				float4 tmp = final_output_sum[g][f];
				tmp.x = 0; tmp.y = 0; tmp.z = 0; tmp.w = 0;
				final_output_sum[g][f] = tmp;
			}
		}

		for (int w = 0; w < NUM_WAPRS; ++w) {
			for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
				for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT; ++f) {
					float4 tmp_in = output_sum[w][g][f];
					float4 tmp_out = final_output_sum[g][f];

					if (SHOULD_PLOT(1)) {
						printf("(in) %f, ",tmp_in.x);
						printf("%f, ",tmp_in.y);
						printf("%f, ",tmp_in.z);
						printf("%f, ",tmp_in.w);
						printf("\n");
						printf("(out) %f, ",tmp_out.x);
						printf("%f, ",tmp_out.y);
						printf("%f, ",tmp_out.z);
						printf("%f, ",tmp_out.w);
						printf("\n");
					}


					tmp_out.x += tmp_in.x;
					tmp_out.y += tmp_in.y;
					tmp_out.z += tmp_in.z;
					tmp_out.w += tmp_in.w;
					final_output_sum[g][f] = tmp_out;
				}
			}
		}
		if (SHOULD_PLOT(1)) {
			printf("final output sum:\n");
			for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
				for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT; ++f) {
					printf("%f, ",final_output_sum[g][f].x);
					printf("%f, ",final_output_sum[g][f].y);
					printf("%f, ",final_output_sum[g][f].z);
					printf("%f, ",final_output_sum[g][f].w);
					printf("\n");
				}
			}
		}
		// finally, store to appropriate output array
		for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
			for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT; ++f) {

				atomicAdd(&current_output[g*F_/BATCH_FILTERS_STRUCT + f].x, final_output_sum[g][f].x);
				atomicAdd(&current_output[g*F_/BATCH_FILTERS_STRUCT + f].y, final_output_sum[g][f].y);
				atomicAdd(&current_output[g*F_/BATCH_FILTERS_STRUCT + f].z, final_output_sum[g][f].z);
				atomicAdd(&current_output[g*F_/BATCH_FILTERS_STRUCT + f].w, final_output_sum[g][f].w);
			}
		}
	}


#endif
}

/*
 * Current best combination: :
 * 		#define CUDA_THREADS 256
 * 		#define BLOCK_X 32/BATCH_PIXELS_SIZE
 * 		#define BLOCK_Y 16
 * 		#define BATCH_PIXELS_SIZE 4
 * 		#define BATCH_FILTERS_SIZE 4
 * 		#define BATCH_FILTERS_SIZE 4
 * 		#define BATCH_FILTERS_STRUCT 4
 * 		#define NUM_FILTERS_PER_THREAD (BATCH_FILTERS_STRUCT*2)
 * 		#define NUM_GAUSS_PER_THREAD BATCH_FILTERS_SIZE
 * 		#define BATCH_DOT 2
 *      #define BATCH_DOT_STORE 1
 *      - hard limit to reg usage; --maxrregcount=125
 */
template <>
void filterActs_YxX_color<double>(const double* images, const double* error, const double* filters, double* output,
										const int I, const int S, const int F, const int G,
										const int subfeat_i, const int feat_i, const int gauss_i,
										const int img_width, const int img_height,
										const int error_width, const int error_height,
										const int kernel_width, const int kernel_height,
										const int padding, const int stride, cudaStream_t streamId) {

}

#include <iostream>

dim3 calculateBlocks(dim3 threadsPerBlock, int error_width, int error_height, int S, int F, int G, int BATCH_PIXELS_SIZE_, int NUM_FILTERS_PER_THREAD, int NUM_GAUSS_PER_THREAD) {
	dim3 numBlocks(((int)ceil(error_width/(float)BATCH_PIXELS_SIZE_) + threadsPerBlock.x - 1) / threadsPerBlock.x,	// over image width (N pixels per thread where N=BATCH_PIXELS_SIZE)
			       (error_height + threadsPerBlock.y - 1) / threadsPerBlock.y,					// over image height
			       // over (subfeature,feature,gauss) combination (M filters and K gausses per thread where M=NUM_FILTERS_PER_THREAD and K=NUM_GAUSS_PER_THREAD
			       S * (int)((F + NUM_FILTERS_PER_THREAD -1) /NUM_FILTERS_PER_THREAD) * (int)( (G+NUM_GAUSS_PER_THREAD-1)/NUM_GAUSS_PER_THREAD) );

//	std::cout << "num blocks: " << numBlocks.x << " " << numBlocks.y << " " << numBlocks.z << std::endl;
	return numBlocks;
}

#define DISPATCH_KERNEL_BOUND_CHECK(BLOCK_X,BLOCK_Y, Kx,Ky,NUM_FILTERS_PER_THREAD, NUM_GAUSS_PER_THREAD, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	 { \
		dim3 numBlocks = calculateBlocks(threadsPerBlock, error_width, error_height, S, F, G, BATCH_PIXELS_SIZE, NUM_FILTERS_PER_THREAD, NUM_GAUSS_PER_THREAD); \
		if (error_width % (BLOCK_X*BATCH_PIXELS_SIZE) == 0)  \
			filterActs_YxX_color_kernel<BLOCK_X,BLOCK_Y,Kx,Ky,NUM_FILTERS_PER_THREAD,NUM_GAUSS_PER_THREAD,0><<<numBlocks,threadsPerBlock,resMem,streamId>>>(images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride); \
		else \
			filterActs_YxX_color_kernel<BLOCK_X,BLOCK_Y,Kx,Ky,NUM_FILTERS_PER_THREAD,NUM_GAUSS_PER_THREAD,1><<<numBlocks,threadsPerBlock,resMem,streamId>>>(images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride); \
		CUDA_POST_KERNEL_CHECK; \
     }

//} else if (F % 8 == 0 && BLOCK_Y > 8) {


#define DISPATCH_KERNEL_FILTERS_LARGE(BLOCK_X,BLOCK_Y, Kx,Ky,NUM_GAUSS_PER_THREAD, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	if (F % 32 == 0 && NUM_GAUSS_PER_THREAD == 1) { \
		DISPATCH_KERNEL_BOUND_CHECK(BLOCK_X,BLOCK_Y, Kx,Ky,32, NUM_GAUSS_PER_THREAD,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (F % 16 == 0 && NUM_GAUSS_PER_THREAD == 2) { \
		DISPATCH_KERNEL_BOUND_CHECK(BLOCK_X,BLOCK_Y, Kx,Ky,16, NUM_GAUSS_PER_THREAD,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (F % 8 == 0) { \
		DISPATCH_KERNEL_BOUND_CHECK(BLOCK_X,BLOCK_Y, Kx,Ky,8, NUM_GAUSS_PER_THREAD,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (F % 4 == 0) { \
		DISPATCH_KERNEL_BOUND_CHECK(BLOCK_X,BLOCK_Y, Kx,Ky,4, NUM_GAUSS_PER_THREAD,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else { \
		LOG(ERROR) << "unsupported number of features F=" << F << ", supported a multiply of 4 or 8!";\
		throw std::exception();\
	}

#define DISPATCH_KERNEL_FILTERS(BLOCK_X,BLOCK_Y, Kx,Ky,NUM_GAUSS_PER_THREAD, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	if (F % 8 == 0) { \
		DISPATCH_KERNEL_BOUND_CHECK(BLOCK_X,BLOCK_Y, Kx,Ky,8, NUM_GAUSS_PER_THREAD,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (F % 4 == 0) { \
		DISPATCH_KERNEL_BOUND_CHECK(BLOCK_X,BLOCK_Y, Kx,Ky,4, NUM_GAUSS_PER_THREAD,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else { \
		LOG(ERROR) << "unsupported number of features F=" << F << ", supported a multiply of 4 or 8!";\
		throw std::exception();\
	}


#define DISPATCH_KERNEL_GAUSS(BLOCK_X,BLOCK_Y, Kx,Ky, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	if (G % 4 == 0) { \
		DISPATCH_KERNEL_FILTERS(BLOCK_X,BLOCK_Y, Kx,Ky, 4, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (G % 3 == 0) { \
		DISPATCH_KERNEL_FILTERS(BLOCK_X,BLOCK_Y, Kx,Ky, 3, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (G % 2 == 0) { \
		DISPATCH_KERNEL_FILTERS_LARGE(BLOCK_X,BLOCK_Y, Kx,Ky, 2, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else { \
		DISPATCH_KERNEL_FILTERS_LARGE(BLOCK_X,BLOCK_Y, Kx,Ky, 1, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	}

#define DISPATCH_KERNEL(BLOCK_X,BLOCK_Y, threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	if (kernel_width == 3 && kernel_height == 3) { \
		DISPATCH_KERNEL_GAUSS(BLOCK_X,BLOCK_Y, 3,3,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (kernel_width == 5 && kernel_height == 5) { \
		DISPATCH_KERNEL_GAUSS(BLOCK_X,BLOCK_Y, 5,5,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (kernel_width == 7 && kernel_height == 7) { \
		DISPATCH_KERNEL_GAUSS(BLOCK_X,BLOCK_Y, 7,7,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (kernel_width == 9 && kernel_height == 9) { \
		DISPATCH_KERNEL_GAUSS(BLOCK_X,BLOCK_Y, 9,9,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else if (kernel_width == 11 && kernel_height == 11) { \
		DISPATCH_KERNEL_GAUSS(BLOCK_X,BLOCK_Y, 11,11,threadsPerBlock,resMem,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride) \
	} else { \
		LOG(ERROR) << "unsupported kernel size [" << kernel_width << " x " << kernel_height << "], supported only 3x3, 5x5, 7x7, 9x9, 11x11!";\
		throw std::exception();\
	}


template <>
void filterActs_YxX_color<float>(const float* images, const float* error, const float* filters, float* output,
										const int I, const int S, const int F, const int G,
										const int subfeat_i, const int feat_i, const int gauss_i,
										const int img_width, const int img_height,
										const int error_width, const int error_height,
										const int kernel_width, const int kernel_height,
										const int padding, const int stride, cudaStream_t streamId) {

#define CUDA_THREADS 256
#define BLOCK_X 32/BATCH_PIXELS_SIZE
#define BLOCK_Y 16
//#define BLOCK_X 4/BATCH_PIXELS_SIZE
//#define BLOCK_Y 4


	//std::cout << "kernel [" << kernel_width << " x " << kernel_height << "], num features " << F << ", num guass " << G << " img ["<< img_width <<" x "<< img_height <<"]" << " error ["<< error_width <<" x "<<  error_height <<"]" <<  std::endl;
	/*
	// we want each block to process 512 pixels - split that over rows/columns if there are too many columns then multiple
	if (error_width >= 4*BATCH_PIXELS_SIZE) {
		// all ql, one row will have enough data
		// use BLOCK_X = 8
		// use BLOXK_Y = 16 // == 512/(8*BATCH_PIXELS_SIZE)
		dim3 threadsPerBlock(8, 16,1);
		DISPATCH_KERNEL(8,16, threadsPerBlock,0,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	} else if (error_width > 2*BATCH_PIXELS_SIZE) {
		// use BLOCK_X = 4
		// use BLOXK_Y = 32 // == 512/(4*BATCH_PIXELS_SIZE)
		dim3 threadsPerBlock(4, 32,1);
		DISPATCH_KERNEL(4,32, threadsPerBlock,0,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	} else if (error_width > BATCH_PIXELS_SIZE) {
		// use BLOCK_X = 2
		// use BLOXK_Y = 64 // == 512/(2*BATCH_PIXELS_SIZE)
		dim3 threadsPerBlock(2, 64,1);
		DISPATCH_KERNEL(2, 64, threadsPerBlock,0,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	} else {
		// use BLOCK_X = 1
		// use BLOXK_Y = 128 // == 512/(1*BATCH_PIXELS_SIZE)
		dim3 threadsPerBlock(1, 128,1);
		DISPATCH_KERNEL(1,128, threadsPerBlock,0,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	}*/


	if (error_width * error_height / (BLOCK_X * BATCH_PIXELS_SIZE) >= 16) {

		dim3 threadsPerBlock(BLOCK_X, BLOCK_Y,1);

		DISPATCH_KERNEL(BLOCK_X,BLOCK_Y, threadsPerBlock,0,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);

	} else if (error_width * error_height / (BLOCK_X * BATCH_PIXELS_SIZE) >= 8) {

		dim3 threadsPerBlock(BLOCK_X, 8, 1);

		DISPATCH_KERNEL(BLOCK_X,8, threadsPerBlock,0,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);

	} else {

		dim3 threadsPerBlock(BLOCK_X, 4, 1);

		DISPATCH_KERNEL(BLOCK_X,4, threadsPerBlock,0,streamId, images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);

	}

}

}  // namespace caffe

