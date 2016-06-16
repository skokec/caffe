#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <cmath>

#include "glog/logging.h"

#include "caffe/util/custom_backprop.hpp"
#include "caffe/util/custom_cub.cuh"

namespace caffe {


#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( l*num_k + k) * num_j + j)*num_i + i )

// CUSTOM convolution code for gradient computation
template <int Bx, int By, int Kx, int Ky, int F, int G>
__global__ void filterActs_YxX_color_kernel(const float* images, const float* error, const float* filters, float* output,
                                   const int I, const int S, const int F_, const int G_,
								   const int subfeat_i_, const int feat_i_, const int gauss_i_,
                                   const int img_width, const int img_height,
								   const int error_width, const int error_height,
								   const int kernel_width, const int kernel_height,
								   const int padding, const int stride) {

//#define NUM_DERIVATIVE_FILTERS 4

//#define BATCH_PIXELS_SIZE 8 // good value (<70ms?)
#define BATCH_PIXELS_SIZE 4


//#define BATCH_FILTERS_SIZE 4 	// good value (<70ms?)
#define BATCH_FILTERS_SIZE 4	// (goes over G)
#define BATCH_FILTERS_STRUCT 4 	// 4 == float4, 2 == float2 (goes over F)

#define NUM_FILTERS_PER_THREAD (BATCH_FILTERS_STRUCT*2)
#define NUM_GAUSS_PER_THREAD BATCH_FILTERS_SIZE

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
	const unsigned int F_GRID_DIM = (F + NUM_FILTERS_PER_THREAD -1) /  NUM_FILTERS_PER_THREAD;
	const unsigned int G_GRID_DIM = (G + NUM_GAUSS_PER_THREAD-1)    /  NUM_GAUSS_PER_THREAD;

	// feat_i MUST be multiple of 4 otherwise we cannot use float4 for copying output !!!
	const unsigned int subfeat_i = (blockIdx.z * blockDim.z + threadIdx.z) / (F_GRID_DIM * G_GRID_DIM);
	const unsigned int gauss_feat_i = (blockIdx.z * blockDim.z + threadIdx.z) % (F_GRID_DIM * G_GRID_DIM);
	const unsigned int feat_i = NUM_FILTERS_PER_THREAD *  (int)(gauss_feat_i / G_GRID_DIM);
	const unsigned int gauss_i = NUM_GAUSS_PER_THREAD * (gauss_feat_i % G_GRID_DIM);

	// shared memory
#if BATCH_PIXELS_SIZE % 4 == 0

	#define IMAGE_SH_WIDTH Bx*BATCH_PIXELS_SIZE/4 + Kx
	__shared__ float4 image_sh_align[(By + Ky)][(Bx*BATCH_PIXELS_SIZE/4 + Kx)];

#elif BATCH_PIXELS_SIZE % 2 == 0

	#define IMAGE_SH_WIDTH Bx*BATCH_PIXELS_SIZE/2 + Kx
	__shared__ float2 image_sh_align[(By + Ky)][(Bx*BATCH_PIXELS_SIZE/2 + Kx)];

#else

	#define IMAGE_SH_WIDTH Bx*BATCH_PIXELS_SIZE + Kx
	__shared__ float image_sh_align[(By + Ky)][(Bx*BATCH_PIXELS_SIZE + Kx)];

#endif

	float* image_sh = reinterpret_cast<float*>(image_sh_align);

	__shared__ float4 kernel_sh[Ky][Kx][NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT][NUM_GAUSS_PER_THREAD];
	__shared__ float4 output_sum[NUM_GAUSS_PER_THREAD][NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT];
	__shared__ unsigned int image_sh_kernel_offsets[Ky*Kx+2];

	int offset_x = BATCH_PIXELS_SIZE * (blockIdx.x * blockDim.x + threadIdx.x);
	int offset_y = blockIdx.y * blockDim.y + threadIdx.y;

	// load shared memory

	// let first thread load all kernel values
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

		for (unsigned int j = 0; j < Ky; ++j) {
			for (unsigned int i = 0; i < Kx; ++i) {
				image_sh_kernel_offsets[j*Kx + i] = i + j*(Bx*BATCH_PIXELS_SIZE + Kx);
			}
		}
		for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
			for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT; ++f) {
				output_sum[g][f].x = 0; output_sum[g][f].y = 0; output_sum[g][f].z = 0; output_sum[g][f].w = 0;
			}
		}
	}
	// Specialize WarpReduce for type int
	typedef cub::WarpReduce<float> WarpReduce;

#define WARP_SIZE 32
	// Allocate WarpReduce shared memory for one warp
	__shared__ typename WarpReduce::TempStorage warp_reduce_storage[Bx*By / WARP_SIZE];

	int warp_id = threadIdx.x * threadIdx.y * threadIdx.z / warpSize;
	WarpReduce warp_reduce(warp_reduce_storage[warp_id]);


	// point to specific output and filter
	float* current_filter = (float*)filters + OFFSET(subfeat_i,gauss_i,feat_i,0, S,G,F,Kx*Ky); //( ( (subfeat_i)*G + gauss_i)*F + feat_i)*(Kx*Ky);

	// load kernel but only as much as is needed for first batch - not all threads need to particpate in loading !!!
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		for (unsigned int j = 0; j < Ky; ++j) {
			for (unsigned int i = 0; i < Kx; ++i) {
				for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
					for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD/BATCH_FILTERS_STRUCT; ++f) {
						float4 kernel_tmp;
						kernel_tmp.x = current_filter[ (( g*F + f*BATCH_FILTERS_STRUCT + 0)*Ky + j)*Kx + i];
						kernel_tmp.y = current_filter[ (( g*F + f*BATCH_FILTERS_STRUCT + 1)*Ky + j)*Kx + i];
						kernel_tmp.z = current_filter[ (( g*F + f*BATCH_FILTERS_STRUCT + 2)*Ky + j)*Kx + i];
						kernel_tmp.w = current_filter[ (( g*F + f*BATCH_FILTERS_STRUCT + 3)*Ky + j)*Kx + i];
						kernel_sh[j][i][f][g] = kernel_tmp;
					}
				}
			}
		}
	}
	// go over all images
	for (unsigned int img_i = 0; img_i < I; ++img_i) {

		// iterate x,y offset over non-aligned elements
//		for (unsigned int offset_y = blockIdx.y * blockDim.y + threadIdx.y; offset_y < error_height; offset_y += blockDim.y * gridDim.y) {
//			for (unsigned int offset_x = BATCH_PIXELS_SIZE*(blockIdx.x * blockDim.x + threadIdx.x); offset_x < error_width; offset_x += BATCH_PIXELS_SIZE*blockDim.x * gridDim.x) {
		{{

				// point to specific image
				float* current_image = (float*)images + OFFSET(img_i,subfeat_i,0,0, I,S,img_height,img_width); //( img_i*S + subfeat_i)*(img_width * img_height);

				// move image and error pointer to position for this block
				current_image += (img_width * (offset_y + padding) + (offset_x + padding));

				// load image with apron (poor utilization when big kernels)
//				#pragma unroll
				for (unsigned int j = threadIdx.y - Ky/2; j < threadIdx.y + Ky/2; j+=blockDim.y) {
//					#pragma unroll
					for (unsigned int i = threadIdx.x*BATCH_PIXELS_SIZE - Kx/2; i < threadIdx.x*BATCH_PIXELS_SIZE + Kx/2; i+=blockDim.x*BATCH_PIXELS_SIZE) {
						// current_image already at position for this block
//						#pragma unroll
						if (BATCH_PIXELS_SIZE % 4 == 0) {
							for (unsigned int k = 0; k < BATCH_PIXELS_SIZE/4; ++k) {
								float4 tmp;
								tmp.x = current_image[j * img_width + i*BATCH_PIXELS_SIZE + k*4 + 0];
								tmp.y = current_image[j * img_width + i*BATCH_PIXELS_SIZE + k*4 + 1];
								tmp.z = current_image[j * img_width + i*BATCH_PIXELS_SIZE + k*4 + 2];
								tmp.w = current_image[j * img_width + i*BATCH_PIXELS_SIZE + k*4 + 3];
								reinterpret_cast<float4*>(image_sh)[(j + Ky/2)*IMAGE_SH_WIDTH + i + Kx/2 + k] = tmp;
							}
						} else if (BATCH_PIXELS_SIZE % 2 == 0) {
							for (unsigned int k = 0; k < BATCH_PIXELS_SIZE/2; ++k) {
								float2 tmp;
								tmp.x = current_image[j * img_width + i*BATCH_PIXELS_SIZE + k*2 + 0];
								tmp.y = current_image[j * img_width + i*BATCH_PIXELS_SIZE + k*2 + 1];
								reinterpret_cast<float2*>(image_sh)[(j + Ky/2)*IMAGE_SH_WIDTH + i + Kx/2 + k] = tmp;
							}
						} else {
							for (unsigned int k = 0; k < BATCH_PIXELS_SIZE; ++k) {
								image_sh[(j + Ky/2)*IMAGE_SH_WIDTH + i + Kx/2 + k] = current_image[j * img_width + i*BATCH_PIXELS_SIZE + k];
							}
						}
					}
				}

				// make sure all threads have finished loading memory
				__syncthreads();

#define BATCH_DOT 2
#define BATCH_DOT_STORE 1
				// prepare pointers for image data - hold pointers for two loops (where loop goes over kernel sizes)
				float* img_sh_p = &image_sh[threadIdx.y*IMAGE_SH_WIDTH + BATCH_PIXELS_SIZE*threadIdx.x];
				float* img_sh_p_tmp[BATCH_DOT];

				#pragma unroll
				for (int k = 0; k < BATCH_DOT; ++k) {
					img_sh_p_tmp[k] = img_sh_p + image_sh_kernel_offsets[k];
				}

				const uint error_step = error_height*error_height;
				float* current_error = (float*)error + OFFSET(img_i,feat_i,offset_y,offset_x, I,F,error_height,error_width); //( img_i*F + feat_i)*(error_width * error_height);

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
					// note, BATCH_FILTERS_SIZE == NUM_GAUSS_PER_THREAD while BATCH_FILTERS_STRUCT goes over F
					float4 intermediate_sum[BATCH_PIXELS_SIZE][BATCH_FILTERS_SIZE];
					float4 loaded_kernel_vals[BATCH_DOT][BATCH_FILTERS_SIZE];

					// make sure to start with zeros
					#pragma unroll
					for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
						#pragma unroll
						for (int i = 0; i < BATCH_FILTERS_SIZE; ++i) {
							intermediate_sum[jj][i].x = 0; intermediate_sum[jj][i].y = 0; intermediate_sum[jj][i].z = 0; intermediate_sum[jj][i].w = 0;
						}
					}

					// then pre-start the main loop by loading elements for the first loop
					float img_tmp[BATCH_DOT][BATCH_PIXELS_SIZE];
					{
						int data_loading_k = 0;

						#pragma unroll
						for (int ii = 0; ii < BATCH_PIXELS_SIZE; ++ii) {
							img_tmp[data_loading_k][ii] = img_sh_p_tmp[data_loading_k][ii];
						}

						#pragma unroll
						for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
							loaded_kernel_vals[data_loading_k][ii] = kr_sh_p_tmp[data_loading_k][ii];
						}
					}

					// main loop that goes over kernel and image values (does dot product)
					// should only go to last-1 and ignore last one since Kx*Ky will always be odd (assuming BATCH_DOT=2)
					// we then need to handle odd last element seperately
					for (int ij = 0; ij < Ky*Kx - BATCH_DOT; ij+=BATCH_DOT) {

						#pragma unroll
						for (int k = 0; k < BATCH_DOT; ++k) {
							const int processing_k = k;
							const int data_loading_k = (k + 1) % BATCH_DOT;
							const int pointer_loading_k = (k + 2) % BATCH_DOT;

							// move loading pointers for loading in next loop
							img_sh_p_tmp[pointer_loading_k] = img_sh_p + image_sh_kernel_offsets[ij+ k + 1];
							kr_sh_p_tmp[pointer_loading_k] = kr_sh_p_tmp[data_loading_k] + NUM_FILTERS_PER_THREAD * NUM_GAUSS_PER_THREAD / BATCH_FILTERS_STRUCT;

							// start loading elements for next loop
							{
								for (int ii = 0; ii < BATCH_PIXELS_SIZE; ++ii) {
									img_tmp[data_loading_k][ii] = img_sh_p_tmp[data_loading_k][ii];
								}

								#pragma unroll
								for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
									loaded_kernel_vals[data_loading_k][ii] = kr_sh_p_tmp[data_loading_k][ii];
								}

							}
							// start processing element(s) for current loop
							#pragma unroll
							for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
								#pragma unroll
								for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
									intermediate_sum[jj][ii].x +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].x;
									intermediate_sum[jj][ii].y +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].y;
									intermediate_sum[jj][ii].z +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].z;
									intermediate_sum[jj][ii].w +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].w;
								}
							}

						}
					}

					// load error values a bit sooner then they are needed to hide global memory access latency (usually ~200 cycles in maxwell)
					float error_values[BATCH_FILTERS_STRUCT][BATCH_PIXELS_SIZE];

					const uint error_img_size = error_height*error_width;

					#pragma unroll
					for (int kk = 0; kk < BATCH_FILTERS_STRUCT; ++kk) {
						#pragma unroll
						for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
							error_values[kk][jj] = current_error[jj];
						}
						current_error += error_img_size;
					}

					// process last element as well (should be already loaded in k=0)
					{
						int processing_k = 0;

						#pragma unroll
						for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
							#pragma unroll
							for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
								intermediate_sum[jj][ii].x +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].x;
								intermediate_sum[jj][ii].y +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].y;
								intermediate_sum[jj][ii].z +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].z;
								intermediate_sum[jj][ii].w +=  img_tmp[processing_k][jj] * loaded_kernel_vals[processing_k][ii].w;
							}
						}
					}
					// sum over current warp to avoid concurrent memory writes
					// we can multiply with errors all intermediate_sums
					float4 filter_sums[BATCH_FILTERS_SIZE];
					#pragma unroll
					for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
						filter_sums[ii].x = 0;
						filter_sums[ii].y = 0;
						filter_sums[ii].z = 0;
						filter_sums[ii].w = 0;
						#pragma unroll
						for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
							filter_sums[ii].x += intermediate_sum[jj][ii].x * error_values[jj][ii];
							filter_sums[ii].y += intermediate_sum[jj][ii].y * error_values[jj][ii];
							filter_sums[ii].z += intermediate_sum[jj][ii].z * error_values[jj][ii];
							filter_sums[ii].w += intermediate_sum[jj][ii].w * error_values[jj][ii];
						}
						filter_sums[ii].x = warp_reduce.Sum(filter_sums[ii].x);
						filter_sums[ii].y = warp_reduce.Sum(filter_sums[ii].y);
						filter_sums[ii].z = warp_reduce.Sum(filter_sums[ii].z);
						filter_sums[ii].w = warp_reduce.Sum(filter_sums[ii].w);
					}

					if (threadIdx.x * threadIdx.y * threadIdx.z  % warpSize == 0) {
						#pragma unroll
						for (int i = 0; i < BATCH_FILTERS_SIZE; ++i) {
							// note, NUM_GAUSS_PER_THREAD == BATCH_FILTERS_SIZE
							/*atomicAdd(&output_sum[i][f].x, (filter_sums[i].x));
							atomicAdd(&output_sum[i][f].y, (filter_sums[i].y));
							atomicAdd(&output_sum[i][f].z, (filter_sums[i].z));
							atomicAdd(&output_sum[i][f].w, (filter_sums[i].w));*/
							float4 tmp = output_sum[i][f];
							tmp.x += filter_sums[i].x;
							tmp.y += filter_sums[i].y;
							tmp.z += filter_sums[i].z;
							tmp.w += filter_sums[i].w;
							output_sum[i][f] = tmp;
						}
					}

					//for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj)
					//	current_error[jj] += BATCH_FILTERS_STRUCT*error_height*error_width;
				}


				// make sure all threads have finished computation before loading next image
				__syncthreads();
			}
		}
	}

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		// NOTE: must be sync from before !!

		// feat_i MUST be multiple of 4 !!!
		float4* current_output = reinterpret_cast<float4*>((float*)output + OFFSET(0,subfeat_i,gauss_i,feat_i, 1,S,G,F)); //( (  subfeat_i * G + gauss_i)*F + feat_i )

		// if unrolling it will use F registers (use F_ to prevent unrolling)
		for (unsigned int g = 0; g < NUM_GAUSS_PER_THREAD; ++g) {
			for (unsigned int f = 0; f < NUM_FILTERS_PER_THREAD / BATCH_FILTERS_STRUCT; ++f) {
				current_output[g*F/BATCH_FILTERS_STRUCT + f] = output_sum[g][f];
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
	dim3 threadsPerBlock(BLOCK_X, BLOCK_Y,1);

	dim3 numBlocks((error_width/BATCH_PIXELS_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,	// over image width (N pixels per thread where N=BATCH_PIXELS_SIZE)
				   (error_height + threadsPerBlock.y - 1) / threadsPerBlock.y,					// over image height
				   // over (subfeature,feature,gauss) combination (M filters and K gausses per thread where M=NUM_FILTERS_PER_THREAD and K=NUM_GAUSS_PER_THREAD
				   S * (int)((F + NUM_FILTERS_PER_THREAD -1) /NUM_FILTERS_PER_THREAD) * (int)( (G+NUM_GAUSS_PER_THREAD-1)/NUM_GAUSS_PER_THREAD) );

	LOG(INFO) << "running filterActs_YxX_color";
	if (kernel_width == 3 && kernel_height == 3 && F == 32 && G == 4)
		filterActs_YxX_color_kernel<BLOCK_X,BLOCK_Y,3,3,32,4><<<numBlocks,threadsPerBlock,0,streamId>>>(images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	else if (kernel_width == 5 && kernel_height == 5 && F == 32 && G == 4)
		filterActs_YxX_color_kernel<BLOCK_X,BLOCK_Y,5,5,32,4><<<numBlocks,threadsPerBlock,0,streamId>>>(images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	else if (kernel_width == 7 && kernel_height == 7 && F == 32 && G == 4)
		filterActs_YxX_color_kernel<BLOCK_X,BLOCK_Y,7,7,32,4><<<numBlocks,threadsPerBlock,0,streamId>>>(images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	else if (kernel_width == 9 && kernel_height == 9 && F == 32 && G == 4)
		filterActs_YxX_color_kernel<BLOCK_X,BLOCK_Y,9,9,32,4><<<numBlocks,threadsPerBlock,0,streamId>>>(images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	else if (kernel_width == 11 && kernel_height == 11 && F == 32 && G == 4)
		filterActs_YxX_color_kernel<BLOCK_X,BLOCK_Y,11,11,32,4><<<numBlocks,threadsPerBlock,0,streamId>>>(images, error, filters, output, I, S, F, G, subfeat_i, feat_i, gauss_i, img_width, img_height, error_width, error_height, kernel_width, kernel_height, padding, stride);
	else
		LOG(INFO) << "no predefined kernel for filterActs_YxX_color";
}

}  // namespace caffe

// OLD VERSIONS
// perform convolution
/*float4 sum_values[F];
#pragma unroll
for (int j = 0; j < Ky; ++j) {

	int i = 0;
	#pragma unroll
	for (i = 0; i < Kx; i+=4) {
		#pragma unroll
		for (int f = 0; f < F; ++f) {
			sum_values[f].x += image_sh[threadIdx.y + j][threadIdx.x + i] *  kernel_sh[f][j][i];
			sum_values[f].y += image_sh[threadIdx.y + j][threadIdx.x + i+1] *  kernel_sh[f][j][i+1];
			sum_values[f].z += image_sh[threadIdx.y + j][threadIdx.x + i+2] *  kernel_sh[f][j][i+2];
			sum_values[f].w += image_sh[threadIdx.y + j][threadIdx.x + i+3] *  kernel_sh[f][j][i+3];
		}
	}
	for (; i < Kx; ++i) {
		for (int f = 0; f < F; ++f) {
			sum_values[f].x += image_sh[threadIdx.y + j][threadIdx.x + i] *  kernel_sh[f][j][i];
		}
	}
}*/
/*
 *
 *
float intermediate_sum[F];
for (int f = 0; f < F; ++f) {
	intermediate_sum[f] = 0;
}

//				#pragma unroll
for (int j = 0; j < Ky; ++j) {
//					#pragma unroll
	for (int i = 0; i < Kx; ++i) {
		float img_tmp = image_sh[threadIdx.y + j][threadIdx.x + i];

		for (int f = 0; f < F; ++f) {
			intermediate_sum[f] +=  img_tmp * kernel_sh[j][i][f];
		}
	}
}

// and multiply with error
#pragma unroll
for (int f = 0; f < F; ++f) {

	float* current_error = (float*)error + OFFSET(img_i,f,0,0, I,F,error_height,error_width); //( img_i*F + feat_i)*(error_width * error_height);
	current_error += (error_width * offset_y + offset_x);

	// TODO: sum over current warp to reduce shared memory usage
	output_sum[f] += current_error[threadIdx.y * error_width + threadIdx.x] * (intermediate_sum[f]);
}
// make sure all threads have finished computation before loading next image
__syncthreads();*/

/*
float* current_error[BATCH_PIXELS_SIZE];
for (int i = 0; i < BATCH_PIXELS_SIZE; ++i)
	current_error[i] = (float*)error + OFFSET(img_i,0,offset_y,offset_x+i, I,F,error_height,error_width); //( img_i*F + feat_i)*(error_width * error_height);


// and multiply with error
//				#pragma unroll
for (int f = 0; f < F_; f+=4*BATCH_FILTERS_SIZE) {
//for (int f = 0; f < F; ++f) {
//{ int f = 0;

	float4 intermediate_sum[BATCH_PIXELS_SIZE][BATCH_FILTERS_SIZE];
	float4 loaded_val[BATCH_FILTERS_SIZE];

	#pragma unroll
	for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
		#pragma unroll
		for (int i = 0; i < BATCH_FILTERS_SIZE; ++i) {
			intermediate_sum[jj][i].x = 0;
			intermediate_sum[jj][i].y = 0;
			intermediate_sum[jj][i].z = 0;
			intermediate_sum[jj][i].w = 0;
		}
	}
	float* img_sh_p = &image_sh[threadIdx.y][BATCH_PIXELS_SIZE*threadIdx.x];
	float* kr_sh_p = &kernel_sh[0][0][f];

//					#pragma unroll
	for (int j = 0; j < Ky; ++j) {
7//						#pragma unroll
		for (int i = 0; i < Kx; ++i) {
	 //{{int i = 1; int j = 2;

			//float img_tmp = image_sh[threadIdx.y + j][threadIdx.x + i];
			//float* img_tmp = img_sh_p;

			float img_tmp[BATCH_PIXELS_SIZE];
			#pragma unroll
			for (int ii = 0; ii < BATCH_PIXELS_SIZE; ++ii) {
				img_tmp[ii] = img_sh_p[ii];
			}

			#pragma unroll
			for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
				//loaded_val[ii].x = kernel_sh[j][i][f + BATCH_FILTERS_SIZE*ii + 0];
				loaded_val[ii].x = kr_sh_p[4*ii + 0];
				loaded_val[ii].y = kr_sh_p[4*ii + 1];
				loaded_val[ii].z = kr_sh_p[4*ii + 2];
				loaded_val[ii].w = kr_sh_p[4*ii + 3];
			}

			img_sh_p+=BATCH_PIXELS_SIZE;
			kr_sh_p+=F;

			#pragma unroll
			for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
				#pragma unroll
				for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
					intermediate_sum[jj][ii].x +=  img_tmp[jj] * loaded_val[ii].x;
					intermediate_sum[jj][ii].y +=  img_tmp[jj] * loaded_val[ii].y;
					intermediate_sum[jj][ii].z +=  img_tmp[jj] * loaded_val[ii].z;
					intermediate_sum[jj][ii].w +=  img_tmp[jj] * loaded_val[ii].w;
				}
			}
		}
		img_sh_p+= Bx;
	}


	// TODO: sum over current warp to reduce shared memory usage
	#pragma unroll
	for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj) {
		#pragma unroll
		for (int i = 0; i < BATCH_FILTERS_SIZE; ++i) {
			output_sum[f+0 +4*i] += current_error[jj][error_offsets[i].x] * (intermediate_sum[jj][i].x);
			output_sum[f+1 +4*i] += current_error[jj][error_offsets[i].y] * (intermediate_sum[jj][i].y);
			output_sum[f+2 +4*i] += current_error[jj][error_offsets[i].z] * (intermediate_sum[jj][i].z);
			output_sum[f+3 +4*i] += current_error[jj][error_offsets[i].w] * (intermediate_sum[jj][i].w);
		}
	}
	for (int jj = 0; jj < BATCH_PIXELS_SIZE; ++jj)
		current_error[jj] += 4*BATCH_FILTERS_SIZE*error_height*error_width;

}
// make sure all threads have finished computation before loading next image
__syncthreads();*/


////////////////

/*
							{
								#define BACTHED_MUL_ADD(C, A,B,k, ii, jj) \
									C[jj][ii].x +=  A[k][jj] * B[k][ii].x; \
									C[jj][ii].y +=  A[k][jj] * B[k][ii].y; \
									C[jj][ii].z +=  A[k][jj] * B[k][ii].z; \
									C[jj][ii].w +=  A[k][jj] * B[k][ii].w;

								// first loop to also issue loading instructions for kernel (for next one)
								#pragma unroll
								for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
									loaded_val[data_loading_k][ii] = kr_sh_p_tmp[data_loading_k][ii];
									BACTHED_MUL_ADD(intermediate_sum, img_tmp, loaded_val, processing_k, ii, 0);
								}
								// second loop to also issue loading instructions for image (for next one)
								img_tmp[data_loading_k][0] = img_sh_p_tmp[data_loading_k][0];
								#pragma unroll
								for (int jj = 1; jj < BATCH_PIXELS_SIZE; ++jj) {
									img_tmp[data_loading_k][jj] = img_sh_p_tmp[data_loading_k][jj];
									BACTHED_MUL_ADD(intermediate_sum, img_tmp, loaded_val, processing_k, 0, jj);
								}
								#pragma unroll
								for (int ii = 1; ii < BATCH_FILTERS_SIZE; ++ii) {
									#pragma unroll
									for (int jj = 1; jj < BATCH_PIXELS_SIZE; ++jj) {
										BACTHED_MUL_ADD(intermediate_sum, img_tmp, loaded_val, processing_k, ii, jj);
									}
								}
							}
							//__syncthreads();
							{
								int k = 1;
								const int processing_k = k;
								const int data_loading_k = (k + 1) % BATCH_DOT;
								const int pointer_loading_k = (k + 2) % BATCH_DOT;

								// move loading pointers for loading in next loop
								img_sh_p_tmp[pointer_loading_k] = img_sh_p + image_sh_kernel_offsets[ij+ k + 1];
								kr_sh_p_tmp[pointer_loading_k] = kr_sh_p_tmp[data_loading_k] + F/BATCH_FILTERS_STRUCT;
								{
									// first loop to also issue loading instructions for kernel (for next one)
									#pragma unroll
									for (int ii = 0; ii < BATCH_FILTERS_SIZE; ++ii) {
										loaded_val[data_loading_k][ii] = kr_sh_p_tmp[data_loading_k][ii];
										BACTHED_MUL_ADD(intermediate_sum, img_tmp, loaded_val, processing_k, ii, 0);
									}
									// second loop to also issue loading instructions for image (for next one)
									img_tmp[data_loading_k][0] = img_sh_p_tmp[data_loading_k][0];
									#pragma unroll
									for (int jj = 1; jj < BATCH_PIXELS_SIZE; ++jj) {
										img_tmp[data_loading_k][jj] = img_sh_p_tmp[data_loading_k][jj];
										BACTHED_MUL_ADD(intermediate_sum, img_tmp, loaded_val, processing_k, 0, jj);
									}
									#pragma unroll
									for (int ii = 1; ii < BATCH_FILTERS_SIZE; ++ii) {
										#pragma unroll
										for (int jj = 1; jj < BATCH_PIXELS_SIZE; ++jj) {
											BACTHED_MUL_ADD(intermediate_sum, img_tmp, loaded_val, processing_k, ii, jj);
										}
									}
								}
							}*/

