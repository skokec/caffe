#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <cmath>
#include <device_launch_parameters.h>

#include "glog/logging.h"

#include "caffe/util/fast_gauss_backward.hpp"
#include "caffe/util/custom_cub.cuh"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// TODO: using hardcoded warp size may not be portable (should use warpSize) but this way allows compiler optimization and avoids using dynamic memory allocation
#define WARP_SIZE 32

#define MAX(x,y) (x > y ? x : y)

#define OFFSET(l,k,j,i, num_l, num_k, num_j, num_i) ((( (l)*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )
#define OFFSET5(m, l,k,j,i, num_m, num_l, num_k, num_j, num_i) ((( ((m)*(num_l) + (l))*(num_k) + (k)) * (num_j) + (j))*(num_i) + (i) )

#define IS_VALID_PIXEL(X,Y,MAX_X,MAX_Y) (X >= 0 && X < MAX_X && Y >= 0 && Y < MAX_Y)

struct  __builtin_align__(16) ptr4
{
	float* quad[4];
};


template <int _NUM_SM,
			int _Bx, int _By,
			int _BLOCK_FEATURES,
            int _BLOCK_SUBFEATURES,
			int _BATCH_PIXELS_SIZE_X,
			int _BATCH_PIXELS_SIZE_Y,
			int _PIXELS_INTERPOLATION_Dx,
			int _PIXELS_INTERPOLATION_Dy,
			int _BATCH_FEATURES_SIZE,
			int _BATCH_MEM_SUBFEATURES_SIZE,
			int _BATCH_GAUSS_SIZE,
			int _IMG_WIDTH, int _IMG_HEIGHT,
			int _MAX_OFFSET>
class BlockIndexing {
public:

	enum {
		NUM_SM = _NUM_SM,
		Bx = _Bx,
		By = _By,
		BLOCK_FEATURES = _BLOCK_FEATURES,
        BLOCK_SUBFEATURES = _BLOCK_SUBFEATURES,
		BATCH_PIXELS_SIZE_X = _BATCH_PIXELS_SIZE_X,
		BATCH_PIXELS_SIZE_Y = _BATCH_PIXELS_SIZE_Y,
		PIXELS_INTERPOLATION_Dx = _PIXELS_INTERPOLATION_Dx,
		PIXELS_INTERPOLATION_Dy = _PIXELS_INTERPOLATION_Dy,
		BATCH_FEATURES_SIZE = _BATCH_FEATURES_SIZE,
		BATCH_MEM_SUBFEATURES_SIZE = _BATCH_MEM_SUBFEATURES_SIZE,
		BATCH_GAUSS_SIZE = _BATCH_GAUSS_SIZE,
		IMG_WIDTH = _IMG_WIDTH,
		IMG_HEIGHT = _IMG_HEIGHT,
		MAX_OFFSET = _MAX_OFFSET,
		NUM_THREADS = Bx* By * BLOCK_FEATURES,
        NUM_WARPS = Bx*By*BLOCK_FEATURES >= WARP_SIZE ? ((Bx*By*BLOCK_FEATURES) / WARP_SIZE) : 1
	};

	// CPU only functions
	class Launch {
	public:
		dim3 getThreadsPerBlock(int num_images, int num_features, int num_subfeatures, int img_width, int img_height) {
			// number of threads per blocks
			return dim3(Bx * By * BLOCK_FEATURES, 1, 1);
            // only BLOCK_FEATURES are distributed over threads while BLOCK_SUBFEATURES are iterated over inside of each thread
		}

		dim3 getBlocksPerGrid(int num_images, int num_features, int num_subfeatures, int img_width, int img_height) {

			// number of blocks per kernel launch
			return dim3 ( (int)ceil(num_features/(BLOCK_FEATURES * BATCH_FEATURES_SIZE)),
                          (int)ceil(num_subfeatures/(BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE)),
						(int)ceil(img_width /  (float)(Bx * BATCH_PIXELS_SIZE_X) ) * (int)ceil(img_height / (float)(By * BATCH_PIXELS_SIZE_Y) ));
		}

	};

	// GPU only functions
	class Kernel {
	public:
		int2 img_size;

		int f_thread_idx;
		int px_thread_idx;

		int img_block_idx;
		int f_block_idx;
        int s_block_idx;

		__device__ Kernel(int img_width, int img_height) {
			img_size.x = img_width;
			img_size.y = img_height;

			f_thread_idx = threadIdx.x / (Bx * By);
			px_thread_idx = threadIdx.x % (Bx * By);

			f_block_idx = blockIdx.x;
            s_block_idx = blockIdx.y;
		}
		// return global image index that specific thread handles
		__device__ int getImageIdx() {
			//return blockIdx.z * NUM_SM  + img_block_idx;
            return 0;
		}

		// return global feature index that specific thread handles
		// since each thread handles multiple features (BATCH_FEATURES_SIZE) and each block handles
		// multiple features as well (BLOCK_FEATURES) this returns offset to F that specific thread will use
		__device__ int getFeatureIdx() {
			return f_block_idx * (BLOCK_FEATURES * BATCH_FEATURES_SIZE)  + f_thread_idx * BATCH_FEATURES_SIZE;
		}

		// return local index that specific thread handles
		// since one block handles multiple feature (BLOCK_FEATURES) this returns index of feature for within one block
		__device__ int getFeatureBlockIdx() {
			return f_thread_idx * BATCH_FEATURES_SIZE;
		}

		__device__ int getSubfeatureIdx() {
            return s_block_idx * (BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE)  + 0;
		}

		__device__ int2 getPosBlockSize() {
			return make_int2(Bx * BATCH_PIXELS_SIZE_X,
							 By * BATCH_PIXELS_SIZE_Y);
		}

		__device__ int2 getPosBlockIdx() {

			int blockIdx_x = blockIdx.z % (img_size.x / (Bx * BATCH_PIXELS_SIZE_X));
			int blockIdx_y = blockIdx.z / (img_size.x / (Bx * BATCH_PIXELS_SIZE_X));

			return make_int2(BATCH_PIXELS_SIZE_X * (blockIdx_x * Bx),
							 BATCH_PIXELS_SIZE_Y * (blockIdx_y * By));
		}

		__device__ int2 getPosThreadIdx() {

			int threadIdx_x = px_thread_idx % (Bx);
			int threadIdx_y = px_thread_idx / (Bx);

			return make_int2(BATCH_PIXELS_SIZE_X * threadIdx_x,
							 BATCH_PIXELS_SIZE_Y * threadIdx_y);
		}

        __device__ int getThreadId() {
            return threadIdx.x +
                   threadIdx.y * blockDim.x +
                   threadIdx.z * blockDim.x * blockDim.y;
        }

        static __device__ int getNumWarps() {
            return NUM_WARPS;
        }
        __device__ int getWarpId() {
            return getThreadId() / warpSize;
        }

        static __forceinline__ __device__ unsigned warp_lane_id()
        {
            unsigned ret;
            asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
            return ret;
        }

        static __forceinline__ __device__ unsigned warp_id()
        {
            unsigned ret;
            asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
            return ret;
        }
	};
};



template <int _NUM_THREADS, int _WIDTH, int _HEIGHT, int _APRON_SIZE, int _NUM_BUFFER_REPEAT, typename _ELEMENT_TYPE, int _BATCH_ELEMENTS>
class BlockSharedMemory {

public:
	typedef _ELEMENT_TYPE ELEMENT_TYPE;

	enum {
		NUM_THREADS = _NUM_THREADS,
		WIDTH = _WIDTH,
		HEIGHT = _HEIGHT,
		APRON_SIZE = _APRON_SIZE,
		NUM_BUFFER_REPEAT = _NUM_BUFFER_REPEAT,
		BATCH_ELEMENTS = _BATCH_ELEMENTS,

		PIXELS_X = WIDTH + 2*APRON_SIZE,
		PIXELS_Y = HEIGHT + 2*APRON_SIZE,

		ALLOC_HEIGHT = PIXELS_Y,

		// we can load N pixels with one LOAD operation so buffer needs to be a multiple of N
		// also make sure to add LOAD for last few pixels
		ALLOC_WIDTH = (PIXELS_X + BATCH_ELEMENTS-1)/BATCH_ELEMENTS,

		// PITCHED_WIDTH == actualy width of allocated data in floats
		PITCHED_WIDTH = ALLOC_WIDTH * BATCH_ELEMENTS,

		// actual size of buffer for each [WIDTH x HEIGHT] patch in number of basic elements (i.e. float)
		// inclues padding to make sure it alights with BATCH_ELEMENTS
		PATCH_SIZE = PITCHED_WIDTH *  ALLOC_HEIGHT,

		// distribute number  of threads per width and height
		// assign consecutive tid to adjecent memory elemnets where each id handled N-elements (N==BATCH_ELEMENTS)
		NUM_THREADS_WIDTH = WIDTH / BATCH_ELEMENTS,
		NUM_THREADS_HEIGHT = NUM_THREADS / NUM_THREADS_WIDTH
	};

private:

	typedef BlockSharedMemory<NUM_THREADS, WIDTH, HEIGHT, APRON_SIZE, NUM_BUFFER_REPEAT, ELEMENT_TYPE, BATCH_ELEMENTS> BlockSharedMemoryT;

	struct _Data {
		ELEMENT_TYPE data[NUM_BUFFER_REPEAT][ALLOC_HEIGHT][ALLOC_WIDTH];
	};

	struct _LoadingData {
		ELEMENT_TYPE data[MAX(1,(HEIGHT + 2*APRON_SIZE) / NUM_THREADS_HEIGHT) ][MAX(1,(WIDTH + 2*APRON_SIZE) / (NUM_THREADS_WIDTH * BATCH_ELEMENTS))];
	};

	float* storage_data_for_writing;
	float* storage_data_for_reading;


	_Data& storage;

	// thread indexing for storing/writing data from global mem
	int2 thread_indexing_writing;

	// thread indexing for reading data by each thread (MUST be user defined in constructor)
	int2 thread_indexing_reading;

public:
	typedef _Data Data;
	typedef _LoadingData LoadingData;

	__device__
	BlockSharedMemory(Data &_storage, int2 read_thread_idx) : storage(_storage), thread_indexing_reading(read_thread_idx) {
		thread_indexing_writing = calcThreadIdx();
		storage_data_for_writing = getDataAt(0, thread_indexing_writing.x/ BATCH_ELEMENTS, thread_indexing_writing.y);
		storage_data_for_reading = getDataAt(0, (thread_indexing_reading.x + APRON_SIZE) / BATCH_ELEMENTS, thread_indexing_reading.y + APRON_SIZE) + (thread_indexing_reading.x + APRON_SIZE) % BATCH_ELEMENTS;
	}

	__device__
	float* getData(int buffer_index = 0) {
		return reinterpret_cast<float*>(storage.data[buffer_index]);
	}
	__device__
	float* getDataThreadIndexingWrite(int buffer_index = 0){
		return storage_data_for_writing + buffer_index * ALLOC_HEIGHT * ALLOC_WIDTH * sizeof(ELEMENT_TYPE) / sizeof(float);
	}

	__device__
	float* getDataThreadIndexingRead(int buffer_index = 0){
		return storage_data_for_reading + buffer_index * ALLOC_HEIGHT * ALLOC_WIDTH * sizeof(ELEMENT_TYPE) / sizeof(float);
	}
	__device__
	float* getDataAt(int index, int x, int y){
		return reinterpret_cast<float*>(&storage.data[index][y][x]);
	}

	template <typename T>
	__device__
	size_t getOffsetAt(int i, int j) {
		return j * ALLOC_WIDTH  * sizeof(ELEMENT_TYPE) / sizeof(T) + i;
	}

	__device__
	int2& getThreadIdx() {
		return this->thread_indexing_writing;
	}

	template <int _GLOBAL_DATA_WIDTH, int REPLICATE_OFFSETED, bool USE_FILL, int FILL_VALUE> // GLOBAL_WIDTH .. size of image row in ELEMENT_TYPE elements i.e. if ELEMENT_TYPE == float4 then GLOBAL_WIDTH counts 4 floats as one
	__device__
	void load_global(const ELEMENT_TYPE* global_data, ELEMENT_TYPE* shared_data, int GLOBAL_DATA_WIDTH = -1, LoadingData *loaded_data = NULL) {

		if (GLOBAL_DATA_WIDTH < 0)
			GLOBAL_DATA_WIDTH = _GLOBAL_DATA_WIDTH;

		// global_data MUST be positioned at [0,0] in global data without APRON, i.e., at [APRON,APRON] / BATCH_ELEMENTS  in shared storage.data
		#pragma unroll
		for (int j = -APRON_SIZE; j < HEIGHT + APRON_SIZE; j+=NUM_THREADS_HEIGHT) {
			#pragma unroll
			for (int i = -APRON_SIZE; i < WIDTH + APRON_SIZE; i+=NUM_THREADS_WIDTH * BATCH_ELEMENTS) {
				// current_image already at position for this block

				if (thread_indexing_writing.x < (WIDTH + APRON_SIZE - i)  && thread_indexing_writing.y < HEIGHT + APRON_SIZE - j)  {

					ELEMENT_TYPE tmp;

					// USING GLOBAL - working
					if (USE_FILL) {
						tmp.x = FILL_VALUE; tmp.y = FILL_VALUE; tmp.z = FILL_VALUE; tmp.w = FILL_VALUE;
					} else {
						tmp = global_data[j * GLOBAL_DATA_WIDTH / BATCH_ELEMENTS + i / BATCH_ELEMENTS];
					}

					int write_offset = (j + APRON_SIZE) * ALLOC_WIDTH  + (i + APRON_SIZE) / BATCH_ELEMENTS;

					if (loaded_data != NULL)
						loaded_data->data[(j + APRON_SIZE)/NUM_THREADS_HEIGHT][(i + APRON_SIZE) / (NUM_THREADS_WIDTH * BATCH_ELEMENTS)] = tmp;
					else {
						// load to sharred data
						shared_data[write_offset] = tmp;

						// replicate the value several times in an offested manner to enable alinged access using float2 or float4 even for unalinged offsets
						for (int replication_index = 0; replication_index < REPLICATE_OFFSETED; ++replication_index) {
							ELEMENT_TYPE* replication_shared_data = shared_data + (replication_index+1) * ALLOC_HEIGHT*ALLOC_WIDTH;

							reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 0] = tmp.x;
							reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 1] = tmp.x;
							reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 2] = tmp.y;
							reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 3] = tmp.z;
						}
					}
				}
			}
		}
//printf("end global read\n");
//		__syncthreads();

//		print();
//		__syncthreads();
	}


	template <int REPLICATE_OFFSETED>
	__device__
	void store_shared(const LoadingData& loaded_data, ELEMENT_TYPE* shared_data) {

#pragma unroll
		for (int j = -APRON_SIZE; j < HEIGHT + APRON_SIZE; j+=NUM_THREADS_HEIGHT) {
#pragma unroll
			for (int i = -APRON_SIZE; i < WIDTH + APRON_SIZE; i+=NUM_THREADS_WIDTH * BATCH_ELEMENTS) {
				// current_image already at position for this block

				if (thread_indexing_writing.x < (WIDTH + APRON_SIZE - i)  && thread_indexing_writing.y < HEIGHT + APRON_SIZE - j)  {

					int write_offset = (j + APRON_SIZE) * ALLOC_WIDTH  + (i + APRON_SIZE) / BATCH_ELEMENTS;

					ELEMENT_TYPE tmp = loaded_data.data[(j + APRON_SIZE)/NUM_THREADS_HEIGHT][(i + APRON_SIZE) / (NUM_THREADS_WIDTH * BATCH_ELEMENTS)];

					// load to sharred data
					shared_data[write_offset] = tmp;

					// replicate the value several times in an offested manner to enable alinged access using float2 or float4 even for unalinged offsets
					for (int replication_index = 0; replication_index < REPLICATE_OFFSETED; ++replication_index) {
						ELEMENT_TYPE* replication_shared_data = shared_data + (replication_index+1) * ALLOC_HEIGHT*ALLOC_WIDTH;

						reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 0] = tmp.x;
						reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 1] = tmp.x;
						reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 2] = tmp.y;
						reinterpret_cast<float*>(replication_shared_data + write_offset)[-1 * (replication_index+1) + 3] = tmp.z;
					}
				}
			}
		}
	}


	__device__
	void print() {
		//if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.y == 0)
        {
			__syncthreads();

			printf("printing shared memory:\n");

			for (int s = 0; s < NUM_BUFFER_REPEAT; ++s) {
				for (int j = 0; j < ALLOC_HEIGHT; ++j){
					for (int i = 0; i < ALLOC_WIDTH; ++i){
                        ELEMENT_TYPE tmp = storage.data[s][j][i];
						printf("%f %f %f %f ", (float)tmp.x, (float)tmp.y, (float)tmp.z, (float)tmp.w);
					}
					printf("\n");
				}
				printf("\nend of NUM_BUFFER_REPEAT %d\n",s);
			}
			printf("\nend of double buffer\n");

			__syncthreads();
		}
	}

private:
	__device__
	static int2 calcThreadIdx() {
		// thread indexes for using to load shared memory
		// we will load N-pixels at onece using LDG.128 and STS.128 so
		// indexing will account for N-pixels in a row handled by a single thread,
		// however since threadIdx may be partitioned differenty, we now need to re-index it

		int thread_index = (blockDim.y * threadIdx.z + threadIdx.y) * blockDim.x + threadIdx.x;

		int2 new_index;
		new_index.x = (thread_index % (NUM_THREADS_WIDTH)) * BATCH_ELEMENTS; // mod by New_x
		new_index.y = thread_index / (NUM_THREADS_WIDTH ); // mod by New_y

		return new_index;
	}

};

template <int _SIZE>
class NDIndexingZero {
public:
	enum {
		SIZE = _SIZE
	};

	template< int DIM>
	static __device__
	int getIndex(int index) {
		if (DIM == 0)
			return index % SIZE;
		else
			return -1;
	}
	static __device__
	int getElementSize() {
		return SIZE;
	}
};

template <int _SIZE, class _PARENT >
class NDIndexing {
public:
	enum {
		SIZE = _SIZE
	};
	typedef _PARENT PARENT;

	template< int DIM>
	static __device__
	int getIndex(int index) {
		if (DIM > 0)
			return PARENT::getIndex<DIM-1>(index);
		else
			return (index / PARENT::getElementSize()) % SIZE;
	}

	static __device__
	int getElementSize() {
		return SIZE * PARENT::getElementSize();
	}
};



template <int BATCH_PIXELS_SIZE_X,
		int BATCH_PIXELS_SIZE_Y,
		bool BATCH_PIXELS_BY_WIDTH,
		int PIXELS_INTERPOLATION_Dx,
		int PIXELS_INTERPOLATION_Dy,
		int BATCH_FEATURES_SIZE,
		int BATCH_COMPUTE_FEATURES_SIZE,
		int BATCH_MEM_SUBFEATURES_SIZE,
		int BLOCK_FEATURES,
		int IMG_WIDTH, int IMG_HEIGHT, int BATCH_PIXELS_FLOAT4,
		typename  _BlockSharedMemoryT>
class PipelineEngine {

	enum {
		PIXELS_INTERPOLATION_SIZE = PIXELS_INTERPOLATION_Dx * PIXELS_INTERPOLATION_Dy,
	};

	_BlockSharedMemoryT& shared_mem;
public:
	typedef _BlockSharedMemoryT BlockSharedMemoryT;
	typedef typename _BlockSharedMemoryT::ELEMENT_TYPE ELEMENT_TYPE;

	__device__
	PipelineEngine(BlockSharedMemoryT& shared_mem_)
		: shared_mem(shared_mem_) {
	}

	// load offset
	struct {
		bool enabled;
		int4* offset_address;
		float* base_address;
		ptr4* output;
	} load_offset;

	// load data
	struct {
		bool enabled;
		ptr4* address;
		float4* output;
	} load_data;

	// compute
	struct {
		bool enabled;
		float4* errors;
		float4* data;
		float4* output;
	} compute;

	// block
	int block_x;
	int block_y;

	int thread_x;
	int thread_y;

#define COPY_VECTOR4(Y,X) \
{ \
	(Y).x = (X).x; \
	(Y).y = (X).y; \
	(Y).z = (X).z; \
	(Y).w = (X).w; \
}

	__device__
	bool should_run(int current_index, int unit_start_delay, int max_iter) {
		return (current_index - unit_start_delay >= 0 && current_index - unit_start_delay < max_iter  ? true : false );
	}

	__device__
	void execute_step() {

		// load quad of offsets for next one and make it directly into pointer to data
		if (load_offset.enabled) {
			for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/4; ++f_quad_index ) {
				load_offset.output[f_quad_index].quad[0] = (float*)((void*)load_offset.base_address + load_offset.offset_address[f_quad_index].x); // F[0]
				load_offset.output[f_quad_index].quad[1] = (float*)((void*)load_offset.base_address + load_offset.offset_address[f_quad_index].y); // F[1]
				load_offset.output[f_quad_index].quad[2] = (float*)((void*)load_offset.base_address + load_offset.offset_address[f_quad_index].z); // F[2]
				load_offset.output[f_quad_index].quad[3] = (float*)((void*)load_offset.base_address + load_offset.offset_address[f_quad_index].w); // F[3]

                /*if (load_offset.offset_address[f_quad_index].x != 0 ||
                        load_offset.offset_address[f_quad_index].y != 0 ||
                        load_offset.offset_address[f_quad_index].z != 0 ||
                        load_offset.offset_address[f_quad_index].w != 0 ) {
                    printf("invalid offsets %d, %d, %d, %d\n", load_offset.offset_address[f_quad_index].x, load_offset.offset_address[f_quad_index].y, load_offset.offset_address[f_quad_index].z, load_offset.offset_address[f_quad_index].w);
                }*/
			}
		}

		NDIndexing<BATCH_COMPUTE_FEATURES_SIZE,
				NDIndexing<PIXELS_INTERPOLATION_Dy,
						NDIndexing<PIXELS_INTERPOLATION_Dx,
								NDIndexingZero<(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4> > > > indexing;

		#pragma unroll
		for (int i = 0; i < BATCH_COMPUTE_FEATURES_SIZE * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /BATCH_PIXELS_FLOAT4; ++i) {

			// i goes over [BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][BATCH_PIXELS_SIZE_/4] array so get indexes for them manually
			int f = indexing.getIndex<0>(i);
			int interpolation_j = indexing.getIndex<1>(i);
			int interpolation_i = indexing.getIndex<2>(i);
			int px = indexing.getIndex<3>(i);
			// since we store weight and offset into float4/int4 we need a proper index to access array of quad vectors
			int f_quad_index = f/4;

			int px_x = px % (BATCH_PIXELS_BY_WIDTH ? BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 : BATCH_PIXELS_SIZE_X);
			int px_y = px / (BATCH_PIXELS_BY_WIDTH ? BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 : BATCH_PIXELS_SIZE_X);

			// since array batches 4 pixels in float4 then get actual px address by multiplying with 4
			px_x = px_x * (BATCH_PIXELS_BY_WIDTH ?  BATCH_PIXELS_FLOAT4 : 1);
			px_y = px_y * (BATCH_PIXELS_BY_WIDTH ?  1 : BATCH_PIXELS_FLOAT4);

			// load data for next loop
			if (load_data.enabled) {

				int data_address_index = f_quad_index;
				int data_quad_index = f % 4;

				if (BATCH_PIXELS_BY_WIDTH) {
					load_data.output[i] = reinterpret_cast<float4*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH)[0];
				} else {
					if (BATCH_PIXELS_FLOAT4 > 0) load_data.output[i].x = reinterpret_cast<float4*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 0) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;
					if (BATCH_PIXELS_FLOAT4 > 1) load_data.output[i].y = reinterpret_cast<float4*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 1) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;
					if (BATCH_PIXELS_FLOAT4 > 2) load_data.output[i].z = reinterpret_cast<float4*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 2) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;
					if (BATCH_PIXELS_FLOAT4 > 3) load_data.output[i].w = reinterpret_cast<float4*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 3) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;

				}

			}

			// compute for current loop
			if (compute.enabled) {
				/*if (f == 0) {
					printf("value at index (%d+%d,%d+%d), %d is (%f) loaded from address %d\n", block_y,thread_y + px_y + 0, block_x, thread_x + px_x, px_y, compute.data[i].x, load_data.address[f_quad_index].quad[f % 4]);
					if (BATCH_PIXELS_FLOAT4 > 1) printf("value at index (%d+%d,%d+%d), %d is (%f) loaded from address %d\n", block_y,thread_y + px_y + 1, block_x, thread_x + px_x, px_y, compute.data[i].y, load_data.address[f_quad_index].quad[f % 4]);
					if (BATCH_PIXELS_FLOAT4 > 2) printf("value at index (%d+%d,%d+%d), %d is (%f) loaded from address %d\n", block_y,thread_y + px_y + 2, block_x, thread_x + px_x, px_y, compute.data[i].z, load_data.address[f_quad_index].quad[f % 4]);
					if (BATCH_PIXELS_FLOAT4 > 3) printf("value at index (%d+%d,%d+%d), %d is (%f) loaded from address %d\n", block_y,thread_y + px_y + 3, block_x ,thread_x + px_x, px_y, compute.data[i].w, load_data.address[f_quad_index].quad[f % 4]);
				}
				*/
				float computed_value = compute.errors[i].x * compute.data[i].x;
				if (BATCH_PIXELS_FLOAT4 > 1) computed_value += compute.errors[i].y * compute.data[i].y;
				if (BATCH_PIXELS_FLOAT4 > 2) computed_value += compute.errors[i].z * compute.data[i].z;
				if (BATCH_PIXELS_FLOAT4 > 3) computed_value += compute.errors[i].w * compute.data[i].w;

                if (f % 4 == 0)
                    compute.output[f_quad_index].x += computed_value;
                else if (f % 4 == 1)
                    compute.output[f_quad_index].y += computed_value;
                else if (f % 4 == 2)
                    compute.output[f_quad_index].z += computed_value;
                else if (f % 4 == 3)
                    compute.output[f_quad_index].w += computed_value;

                //if (threadIdx.x == 0)
                //    printf("thread val: %f (index %d) with = %f * %f (index %d)\n", computed_value, f, compute.errors[i].x , compute.data[i].x, i);

			}
		}
	}

};



template <typename BlockIndexingT>
__global__ void //__launch_bounds__(128, 4)
fast_gauss_backward_pipeline_kernel(cudaTextureObject_t* filtered_images_tex, const float* filtered_images, const float* error_images,
                                    const int* filter_offsets, const float* filter_weights, float* output,
							        const int I, const int S, const int F, const int G,
							        const int img_width_, const int img_height_,
							        const int kernel_width, const int kernel_height) {

// INPUT: filtered images  	[I x S x H x W]
//        error images  	[I x S x H x W]
//		  filter offsets   	[F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES)] x [S / (BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES)] x [G / BATCH_GAUSS_SIZE]
// 				 	            x [ BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES] x [BATCH_GAUSS_SIZE] x [BATCH_FEATURES_SIZE/4] x [BLOCK_FEATURES];
// OUTPUT output  		 	[S x G x F]

#ifndef CUBIN_EMBEDDING

	typedef class BlockIndexingT::Kernel BlockIndexingKernel;

	static const int NUM_SM = BlockIndexingT::NUM_SM;
	static const int Bx = BlockIndexingT::Bx;
	static const int By = BlockIndexingT::By;
	static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
    static const int BLOCK_SUBFEATURES = BlockIndexingT::BLOCK_SUBFEATURES;
	static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
	static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
	static const int PIXELS_INTERPOLATION_Dx = BlockIndexingT::PIXELS_INTERPOLATION_Dx;
	static const int PIXELS_INTERPOLATION_Dy = BlockIndexingT::PIXELS_INTERPOLATION_Dy;
	static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
	static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
	static const int BATCH_GAUSS_SIZE = BlockIndexingT::BATCH_GAUSS_SIZE;
	static const int IMG_WIDTH = BlockIndexingT::IMG_WIDTH;
	static const int IMG_HEIGHT = BlockIndexingT::IMG_HEIGHT; // may not be needed
	static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;

	static const int NUM_THREADS = BlockIndexingT::NUM_THREADS;

	// since we can load 4 offsets from a single LDS.128 we can batch 4 computes of features
	static const int BATCH_COMPUTE_FEATURES_SIZE = 4;

	// using float4 to load so use
	static const int BATCH_SH_PIXELS_SIZE = 4;

	// using float4 for computing pixels
	static const int BATCH_PIXELS_FLOAT4 = 1;

	static const bool BATCH_PIXELS_BY_WIDTH = BATCH_PIXELS_SIZE_X >= 4 && BATCH_PIXELS_FLOAT4 == 4;

	static const int DOUBLE_BUFFERING = 2;

	static const int NUM_REPLICATE_OFFSETED = 1;
    //static const int NUM_REPLICATE_OFFSETED = 0;

	static const int PIXELS_INTERPOLATION_SIZE = PIXELS_INTERPOLATION_Dx * PIXELS_INTERPOLATION_Dy;

	float4* output_batch = reinterpret_cast<float4*>(output);

	int img_width = IMG_WIDTH; //img_width_; //IMG_WIDTH;
	int img_height = IMG_HEIGHT; //img_height_; //IMG_HEIGHT;

	BlockIndexingKernel block_indexing(img_width, img_height);

	//int n = block_indexing.getImageIdx();

	int f_offset = block_indexing.getFeatureIdx();

    int f_block_idx = block_indexing.getFeatureBlockIdx();

    int s_offset = block_indexing.getSubfeatureIdx();

	int block_width = block_indexing.getPosBlockSize().x;
	int block_height = block_indexing.getPosBlockSize().y;

	int block_x = block_indexing.getPosBlockIdx().x;
	int block_y = block_indexing.getPosBlockIdx().y;

	int thread_x = block_indexing.getPosThreadIdx().x;
	int thread_y = block_indexing.getPosThreadIdx().y;

	int G_MEM_SIZE = G / BATCH_GAUSS_SIZE;
	int S_MEM_SIZE = S / ( BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES);
	int F_MEM_SIZE = F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES);

    static const int OFFSET_BLOCK_MEM_SIZE = BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE *  BLOCK_FEATURES;

	typedef BlockSharedMemory<NUM_THREADS,
								Bx * BATCH_PIXELS_SIZE_X,
								By * BATCH_PIXELS_SIZE_Y,
								MAX_OFFSET,
								(NUM_REPLICATE_OFFSETED+1) * DOUBLE_BUFFERING * BATCH_MEM_SUBFEATURES_SIZE,
								float4,
								BATCH_SH_PIXELS_SIZE> SharedMem;

	__shared__ typename SharedMem::Data data;

	SharedMem image_sh_class(data, make_int2(thread_x, thread_y));

	int thread_sh_x = image_sh_class.getThreadIdx().x;
	int thread_sh_y = image_sh_class.getThreadIdx().y;

    static const int DOUBLE_BUFFERING_OFF = 1; // disable double buffering for offsets since we can load all of them before first loop

	typedef BlockSharedMemory<NUM_THREADS, BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE * BLOCK_FEATURES,
								1, 0, DOUBLE_BUFFERING_OFF, int4, BATCH_SH_PIXELS_SIZE> SharedMemOffsets;

#define ENABLE_SHARED_MEM_FOR_OFFSETS 1

#ifdef ENABLE_SHARED_MEM_FOR_OFFSETS
	__shared__ typename SharedMemOffsets::Data data_offsets;

	SharedMemOffsets offsets_sh_class(data_offsets, make_int2(thread_x, thread_y));

	int4* offset_batch_sh = (int4*)offsets_sh_class.getData(0);
#endif


	// WARNING: leave this part in otherwise it works slower !!! (probably due to some compiler optimization)
//	if (threadIdx.x == 100000000)
//		for (int i = 0; i < 2 * BATCH_MEM_SUBFEATURES_SIZE; ++i)
//			image_sh_class.template load_global<IMG_WIDTH + 2*MAX_OFFSET,0,true,0>(reinterpret_cast<float4*>(NULL), reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(i)) ) ;

	//__syncthreads();

    float4 out_val[BATCH_GAUSS_SIZE][BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES][BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE];

    #pragma unroll
    for (int g = 0; g < BATCH_GAUSS_SIZE; ++g) {
        #pragma unroll
        for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES; ++s) {
            #pragma unroll
            for (int f = 0; f < BATCH_FEATURES_SIZE / BATCH_COMPUTE_FEATURES_SIZE; ++f) {
                out_val[g][s][f].x = 0;
                out_val[g][s][f].y = 0;
                out_val[g][s][f].z = 0;
                out_val[g][s][f].w = 0;
            }
        }
    }

	PipelineEngine<BATCH_PIXELS_SIZE_X,
					BATCH_PIXELS_SIZE_Y,
					BATCH_PIXELS_BY_WIDTH,
					PIXELS_INTERPOLATION_Dx,
					PIXELS_INTERPOLATION_Dy,
					BATCH_FEATURES_SIZE,
					BATCH_COMPUTE_FEATURES_SIZE,
					BATCH_MEM_SUBFEATURES_SIZE,
					BLOCK_FEATURES,
					IMG_WIDTH, IMG_HEIGHT, BATCH_PIXELS_FLOAT4,
					SharedMem> pipeline(image_sh_class);

	pipeline.block_x = block_x;
	pipeline.block_y = block_y;
	pipeline.thread_x = thread_x;
	pipeline.thread_y = thread_y;
	const int f_start_block = f_offset - f_block_idx;
/*
    for (int s = 0 ; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {
        int buffer_index = OFFSET(0, 0, s % BATCH_MEM_SUBFEATURES_SIZE, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

        image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,true,1>(reinterpret_cast<const float4*>(0 + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
                                                                                                       reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
                                                                                                       img_width + 2 * MAX_OFFSET);

        buffer_index = OFFSET(0, 1, s % BATCH_MEM_SUBFEATURES_SIZE, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);
        image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,true,1>(reinterpret_cast<const float4*>(0 + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
                                                                                                      reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
                                                                                                      img_width + 2 * MAX_OFFSET);
    }
    offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,true,0>(reinterpret_cast<const int4*>(0 + offsets_sh_class.getThreadIdx().x),
                                                                          reinterpret_cast<int4*>(offsets_sh_class.getDataThreadIndexingWrite(0)));
    offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,true,0>(reinterpret_cast<const int4*>(0 + offsets_sh_class.getThreadIdx().x),
                                                                          reinterpret_cast<int4*>(offsets_sh_class.getDataThreadIndexingWrite(1)));
*/
    const int* _filter_offset_current = filter_offsets +  OFFSET(f_start_block / (BLOCK_FEATURES * BATCH_FEATURES_SIZE),
                                                                 s_offset / (BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE),
                                                                 0,
                                                                 0,
                                                                 F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE);

#ifdef ENABLE_SHARED_MEM_FOR_OFFSETS
    const int* _filter_offset_next = _filter_offset_current + offsets_sh_class.getThreadIdx().x;

    if (1){
        // load offsets for the first one
        offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,false,0>(reinterpret_cast<const int4*>(_filter_offset_current + offsets_sh_class.getThreadIdx().x),
                                                                               reinterpret_cast<int4*>(offsets_sh_class.getDataThreadIndexingWrite(0)));
        __syncthreads();
    }
#else
    const int* _filter_offset_next = _filter_offset_current;
#endif

    const float* _filtered_images = filtered_images + OFFSET(0,
                                                             s_offset,
                                                             MAX_OFFSET + block_y + image_sh_class.getThreadIdx().y,
                                                             MAX_OFFSET + block_x + image_sh_class.getThreadIdx().x,
                                                             I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);

    const float* _error_images = error_images + OFFSET(0,
                                                       f_offset,
                                                       block_y + thread_y,
                                                       block_x + thread_x,
                                                       I, F, img_height, img_width);
    if (1){
        // load first batch of subfeatures/input data into shared memory
        const float* _image_global_current = filtered_images + OFFSET(0,
                                                                      s_offset,
                                                                      MAX_OFFSET + block_y + image_sh_class.getThreadIdx().y,
                                                                      MAX_OFFSET + block_x + image_sh_class.getThreadIdx().x,
                                                                      I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);
        //const float* _image_global_current = _filtered_images;

        for (int s = 0 ; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

            int buffer_index = OFFSET(0, 0, s, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);


            image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
                    //image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,true,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
                                                                                                           reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
                                                                                                           img_width + 2 * MAX_OFFSET);
/*				typename SharedMem::LoadingData ld;
				image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
                                                                                                               reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
                                                                                                               img_width + 2 * MAX_OFFSET, &ld);

				image_sh_class.template store_shared<NUM_REPLICATE_OFFSETED>(ld,reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)));*/
        }
        __syncthreads();
    }
/*
    const int image_offset = S * (img_height + 2*MAX_OFFSET) * (img_width + 2*MAX_OFFSET);
    const int error_offset = F * img_height * img_width;

    float const* _image_global_current = _filtered_images;
    float const* _image_global_next = _filtered_images + image_offset;

    float const* _error_global_current = _error_images;

*/
    for (int n = 0; n < I; ++n) {

        const float* _image_global_current = filtered_images + OFFSET(n,
                                                                      s_offset,
                                                                      MAX_OFFSET + block_y + image_sh_class.getThreadIdx().y,
                                                                      MAX_OFFSET + block_x + image_sh_class.getThreadIdx().x,
                                                                      I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);

		// this is actually the next image
		const float* _image_global_next = filtered_images + OFFSET(n+1,
																	  s_offset,
																	  MAX_OFFSET + block_y + image_sh_class.getThreadIdx().y,
																	  MAX_OFFSET + block_x + image_sh_class.getThreadIdx().x,
																	  I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);
        const float* _error_global_current = error_images + OFFSET(n,
                                                                   f_offset,
                                                                   block_y + thread_y,
                                                                   block_x + thread_x,
                                                                   I, F, img_height, img_width);



		if (0){
			// load first batch of subfeatures/input data into shared memory

			for (int s = 0 ; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

				int buffer_index = OFFSET(0, 0, s, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);


				image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
						//image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,true,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
																											   reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
																											   img_width + 2 * MAX_OFFSET);
/*				typename SharedMem::LoadingData ld;
				image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
                                                                                                               reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
                                                                                                               img_width + 2 * MAX_OFFSET, &ld);

				image_sh_class.template store_shared<NUM_REPLICATE_OFFSETED>(ld,reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)));*/
			}
			__syncthreads();
		}

		// load error values for all the features that are needed in this thread i.e. for BATCH_FEATURES_SIZE (or BATCH_COMPUTE_FEATURES_SIZE ?!!)
        float4 err_vals[BATCH_FEATURES_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4];

		// TODO: loading should be done using LDG.128 or LDG.64 !!!
        #pragma unroll
        for (int ff = 0; ff < BATCH_FEATURES_SIZE; ++ff) {
            if (BATCH_PIXELS_BY_WIDTH) {
                #pragma unroll
                for (int py = 0; py < BATCH_PIXELS_SIZE_Y; ++py) {
                    #pragma unroll
                    for (int px = 0; px < BATCH_PIXELS_SIZE_X; px+=BATCH_PIXELS_FLOAT4) {
                      	/*if (BATCH_PIXELS_FLOAT4 > 0) err_vals[ff][py * BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 + px/BATCH_PIXELS_FLOAT4].x = 1;
                        if (BATCH_PIXELS_FLOAT4 > 1) err_vals[ff][py * BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 + px/BATCH_PIXELS_FLOAT4].y = 1;
                        if (BATCH_PIXELS_FLOAT4 > 2) err_vals[ff][py * BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 + px/BATCH_PIXELS_FLOAT4].z = 1;
                        if (BATCH_PIXELS_FLOAT4 > 3) err_vals[ff][py * BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 + px/BATCH_PIXELS_FLOAT4].w = 1;*/

						if (BATCH_PIXELS_FLOAT4 > 0) err_vals[ff][py * BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 + px/BATCH_PIXELS_FLOAT4].x = _error_global_current[OFFSET(n, ff, py, px + 0, I, F, img_height, img_width)];
						if (BATCH_PIXELS_FLOAT4 > 1) err_vals[ff][py * BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 + px/BATCH_PIXELS_FLOAT4].y = _error_global_current[OFFSET(n, ff, py, px + 1, I, F, img_height, img_width)];
						if (BATCH_PIXELS_FLOAT4 > 2) err_vals[ff][py * BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 + px/BATCH_PIXELS_FLOAT4].z = _error_global_current[OFFSET(n, ff, py, px + 2, I, F, img_height, img_width)];
						if (BATCH_PIXELS_FLOAT4 > 3) err_vals[ff][py * BATCH_PIXELS_SIZE_X/BATCH_PIXELS_FLOAT4 + px/BATCH_PIXELS_FLOAT4].w = _error_global_current[OFFSET(n, ff, py, px + 3, I, F, img_height, img_width)];

                    }
                }
            } else {
                #pragma unroll
                for (int py = 0; py < BATCH_PIXELS_SIZE_Y; py+=BATCH_PIXELS_FLOAT4) {
                    #pragma unroll
                    for (int px = 0; px < BATCH_PIXELS_SIZE_X ; ++px) {
                        /*if (BATCH_PIXELS_FLOAT4 > 0) err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = 1;
                        if (BATCH_PIXELS_FLOAT4 > 1) err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].y = 1;
                        if (BATCH_PIXELS_FLOAT4 > 2) err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].z = 1;
                        if (BATCH_PIXELS_FLOAT4 > 3) err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].w = 1;*/
						const float4 tmp = reinterpret_cast<const float4*>(_error_global_current)[OFFSET(0, ff, py + 0, px/4, I, F, img_height, img_width/4)];

						err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = px % 4 == 0 ? tmp.x :
																							(px % 4 == 1 ? tmp.y :
																							 (px % 4 == 2 ? tmp.z : tmp.w));

						/*if (BATCH_PIXELS_FLOAT4 > 0) err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].x = _error_global_current[OFFSET(0, ff, py + 0, px, I, F, img_height, img_width)];
						if (BATCH_PIXELS_FLOAT4 > 1) err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].y = _error_global_current[OFFSET(0, ff, py + 1, px, I, F, img_height, img_width)];
						if (BATCH_PIXELS_FLOAT4 > 2) err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].z = _error_global_current[OFFSET(0, ff, py + 2, px, I, F, img_height, img_width)];
						if (BATCH_PIXELS_FLOAT4 > 3) err_vals[ff][py/BATCH_PIXELS_FLOAT4 * BATCH_PIXELS_SIZE_X + px].w = _error_global_current[OFFSET(0, ff, py + 3, px, I, F, img_height, img_width)];*/

                    }
                }
            }
        }
        //__syncthreads();

        const int MAX_S_OUTER_INDEX = BLOCK_SUBFEATURES; //S /  BATCH_MEM_SUBFEATURES_SIZE;

        #pragma unroll
        for (int s_outer_index = 0; s_outer_index <  BLOCK_SUBFEATURES; ++s_outer_index) {

            // s_offset_outer is legacy from fast_gauss_forward.cu version
            //const int s_offset_outer = s_offset + s_outer_index * BATCH_MEM_SUBFEATURES_SIZE;

            const int s_buffer_index = s_outer_index % DOUBLE_BUFFERING;

			// next in this filter_offset_next and image_global_next_s_offset means next S (not to be confused with image_global_next_image which is actually the next image)
            const int* filter_offset_current = _filter_offset_current + OFFSET(0, 0, s_outer_index, 0, F_MEM_SIZE,  S_MEM_SIZE * G_MEM_SIZE, BLOCK_SUBFEATURES, OFFSET_BLOCK_MEM_SIZE / BLOCK_SUBFEATURES);
            const int* filter_offset_next = _filter_offset_next + OFFSET(0, 0, s_outer_index + 1, 0, F_MEM_SIZE, S_MEM_SIZE * G_MEM_SIZE, BLOCK_SUBFEATURES, OFFSET_BLOCK_MEM_SIZE / BLOCK_SUBFEATURES);

            const float* image_global_current = _image_global_current + OFFSET(0, s_outer_index, 0, 0, I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);
            const float* image_global_next_s_offset = _image_global_current + OFFSET(0, s_outer_index + 1, 0, 0, I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);

			const float* image_global_next_image = _image_global_next + OFFSET(0, s_outer_index, 0, 0, I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);

            ptr4 off_A[BATCH_GAUSS_SIZE][BATCH_COMPUTE_FEATURES_SIZE/4],
                 off_B[BATCH_GAUSS_SIZE][BATCH_COMPUTE_FEATURES_SIZE/4];

            float4 d_A[BATCH_GAUSS_SIZE][BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4],
                   d_B[BATCH_GAUSS_SIZE][BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/BATCH_PIXELS_FLOAT4];


            struct IterIndex {
                int s; // sub-feature index
                int f; // feature index
                int g; // gauss component index
            };


            // global loading is done imediately (no delay)
            // to simplyfiy the code for global loading we can force global loading to be done BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE loops before
            // other units start
            static const int start_delay_global_load = 0; //1;
            static const int start_delay_offset_load = 0;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
            static const int start_delay_data_load = 1;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
            static const int start_delay_compute = 2;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;

            // NOTE: EXTRA_LOOPS is max value out of start_delay_global_load, start_delay_offset_load, start_delay_w_load, start_delay_data_load and start_delay_compute
            static const int EXTRA_LOOPS = MAX(start_delay_global_load,
                                                MAX(start_delay_offset_load,
                                                     MAX(start_delay_data_load, start_delay_compute)));

            int NUM_ITER = BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;


            // iterations go over subsets of [S x G x F ] i.e. [BATCH_MEM_SUBFEATURES_SIZE] * [BATCH_GAUSS_SIZE] * [BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE]
			typename SharedMem::LoadingData ld[BATCH_MEM_SUBFEATURES_SIZE];

            NDIndexing<BATCH_MEM_SUBFEATURES_SIZE,
                        NDIndexing<BATCH_GAUSS_SIZE,
                            NDIndexingZero<BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE> > > indexing;
            // do all in one loop
            #pragma unroll
            for (int index = 0 ; index < NUM_ITER + EXTRA_LOOPS; ++index)  {

                IterIndex load_global;
                IterIndex load_offset_index;
                IterIndex load_w_index;
                IterIndex load_data_index;
                IterIndex compute_index;

                // get flags to run each unit based on index number and its delay
                pipeline.load_offset.enabled = pipeline.should_run(index, start_delay_offset_load, NUM_ITER);
                pipeline.load_data.enabled = pipeline.should_run(index, start_delay_data_load, NUM_ITER);
                pipeline.compute.enabled = pipeline.should_run(index, start_delay_compute, NUM_ITER);

                bool load_global_enabled = pipeline.should_run(index, start_delay_global_load, NUM_ITER);

                int global_d = -1;
                int shared_d_off = -1;
                int shared_d_current = -1;
                int shared_d_next = -1;
                {
                    // global loading is done immedately
                    load_global.s = indexing.getIndex<0>(index - start_delay_global_load);
                    load_global.g = indexing.getIndex<1>(index - start_delay_global_load);
                    load_global.f = indexing.getIndex<2>(index - start_delay_global_load) * BATCH_COMPUTE_FEATURES_SIZE;

					if (thread_x == 0 && thread_y == 0 && block_x == 0 && block_y == 0 && f_offset == 0 && s_offset == 0) {

						//printf("original global s: %d\n", load_global.s );
					}
                    // we actually load next batch of subfeatures so add BATCH_MEM_SUBFEATURES_SIZE
                    load_global.s = load_global.s + BATCH_MEM_SUBFEATURES_SIZE;

                    if (load_global_enabled)
                        load_global_enabled = load_global.f == 0 && load_global.g == 0;

                    // TODO: if this is last s_outer_index index the load next image (we could also load next errors)

                    int double_buffer_index = (s_buffer_index + load_global.s/BATCH_MEM_SUBFEATURES_SIZE) % DOUBLE_BUFFERING;
                    int subfeat_buffer_index = load_global.s % BATCH_MEM_SUBFEATURES_SIZE;

                    // if this is last iteration before moving to next s_outer_index index then load for the next one
                    bool load_next_s_outer = load_global.s < BATCH_MEM_SUBFEATURES_SIZE ;

					if (thread_x == 0 && thread_y == 0 && block_x == 0 && block_y == 0 && f_offset == 0 && s_offset == 0) {

						//printf("load_global.s: %d, load_next_s_outer s: %d, double_buffer_index %d, subfeat_buffer_index: %d\n", load_global.s, load_next_s_outer,  double_buffer_index, subfeat_buffer_index);
					}

					bool load_next_image = s_outer_index >= MAX_S_OUTER_INDEX - 1 ? true : false;

					//const float* image_global_load =  load_next_s_outer ? image_global_current : image_global_next_s_offset;
					const float* image_global_load =  load_next_image ? image_global_next_image : image_global_next_s_offset;

					if (load_next_image) {

					}

                    load_global.s = load_global.s % (BATCH_MEM_SUBFEATURES_SIZE);

                    // check if this is the last s_outer_index index and skip loading if it is
					// TODO: we could load the next image instead
                    //if (s_outer_index >= MAX_S_OUTER_INDEX - 1)
					if (load_next_image && n >= I-1)
                    	load_global_enabled = false;

                    global_d = double_buffer_index;

                    int buffer_index = OFFSET(0, double_buffer_index, subfeat_buffer_index, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

                    // load global
                    if (load_global_enabled) {

						image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(image_global_load + (load_global.s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
						//image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,true,1>(reinterpret_cast<const float4*>(image_global_load + (load_global.s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
																													   reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
																													   img_width + 2 * MAX_OFFSET);

						/*typename SharedMem::LoadingData ld;
                        image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(image_global_load + (load_global.s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
                                                                                                                       reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
                                                                                                                       img_width + 2 * MAX_OFFSET, &ld);

						image_sh_class.template store_shared<NUM_REPLICATE_OFFSETED>(ld,reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)));*/
                    }

                }
                bool require_sync = false;
                bool load_offset_reg_A;
                {
                    // offset loading is done with no delay

                    load_offset_index.s = indexing.getIndex<0>(index - start_delay_offset_load);
                    load_offset_index.g = indexing.getIndex<1>(index - start_delay_offset_load);
                    load_offset_index.f = indexing.getIndex<2>(index - start_delay_offset_load) * BATCH_COMPUTE_FEATURES_SIZE;

                    int double_buffer_index =   (s_buffer_index + load_offset_index.s/BATCH_MEM_SUBFEATURES_SIZE) % DOUBLE_BUFFERING;
                    int subfeat_buffer_index = load_offset_index.s % BATCH_MEM_SUBFEATURES_SIZE;

                    int next_s = indexing.getIndex<0>((index - start_delay_offset_load) - 1);

                    // enforce sync to ensure data is fully written to shared memory before we will be reading it
                    // we do this each time before we switch to another buffer
                    // note: we do this when load_offset starts which is the first operation so that sync will be nicly segemtnated between batches
                    int s_mem_index = load_offset_index.s >= 0 ? (load_offset_index.s / BATCH_MEM_SUBFEATURES_SIZE) : ((load_offset_index.s + 1) /  BATCH_MEM_SUBFEATURES_SIZE + 1);
                    int s_mem_index_next = next_s >= 0 ? (next_s / BATCH_MEM_SUBFEATURES_SIZE) : ((next_s + 1) /  BATCH_MEM_SUBFEATURES_SIZE + 1);

                    int current_double_buffer_index = (s_buffer_index + s_mem_index) % DOUBLE_BUFFERING;
                    int next_double_buffer_index = (s_buffer_index + s_mem_index_next) % DOUBLE_BUFFERING;

                    if (pipeline.load_offset.enabled) {

                        if (load_offset_index.s >= 0 && load_offset_index.f >= 0 && load_offset_index.g >=0 ) {
                            require_sync = current_double_buffer_index == next_double_buffer_index ? false : true;

                            // handle first loading where prev index goes to negative and will not be handled by the previous line
                            if (load_offset_index.s == 0 && load_offset_index.f == 0 && load_offset_index.g == 0)
                                require_sync = true;

							require_sync = require_sync && (n == 0 && s_outer_index == 0) == false;
                        }
                    }

                    shared_d_next = next_double_buffer_index;

                    // switch between registers every iteration
                    bool use_reg_A = (index - start_delay_offset_load) % 2 == 0 ? true : false;

                    load_offset_reg_A = use_reg_A;

                    shared_d_off = double_buffer_index;

                    int address_off = OFFSET(load_offset_index.s, load_offset_index.g, load_offset_index.f/4, f_block_idx/BATCH_FEATURES_SIZE, BATCH_MEM_SUBFEATURES_SIZE, BATCH_GAUSS_SIZE, BATCH_FEATURES_SIZE/4, BLOCK_FEATURES);

                    // load offset
#ifdef ENABLE_SHARED_MEM_FOR_OFFSETS
                    pipeline.load_offset.offset_address = offset_batch_sh + address_off + (s_outer_index % DOUBLE_BUFFERING_OFF) * OFFSET_BLOCK_MEM_SIZE/4;

#else
                    pipeline.load_offset.offset_address = &reinterpret_cast<int4*>((int*)filter_offset_current)[address_off];
#endif

                    int buffer_index = OFFSET(0, double_buffer_index, subfeat_buffer_index, 0, 1, DOUBLE_BUFFERING_OFF, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

                    pipeline.load_offset.base_address = image_sh_class.getDataThreadIndexingRead(buffer_index);
                    pipeline.load_offset.output = (ptr4*)(use_reg_A ? &off_A[load_offset_index.g][0] : &off_B[load_offset_index.g][0]);


                    /*if (pipeline.load_offset.enabled) {
                        if (thread_x == 0 && thread_y == 0)
                        if (pipeline.load_offset.offset_address[0].x != 0 ||
                                pipeline.load_offset.offset_address[0].y != 0 ||
                                pipeline.load_offset.offset_address[0].z != 0 ||
                                pipeline.load_offset.offset_address[0].w != 0 ) {
                            printf("invalid offsets at index (%d), s,g,f=(%d+%d +%d,%d,%d + %d), values=%d, %d, %d, %d at block/thread(%d+%d, %d + %d)\n", index, s_offset, s_outer_index, load_offset_index.s, load_offset_index.g, f_offset, load_offset_index.f, pipeline.load_offset.offset_address[0].x, pipeline.load_offset.offset_address[0].y, pipeline.load_offset.offset_address[0].z, pipeline.load_offset.offset_address[0].w, block_y, thread_y, block_x, thread_x);
                        }
                    }*/

                }
                bool load_data_reg_A;
                {

                    load_data_index.s = indexing.getIndex<0>(index - start_delay_data_load);
                    load_data_index.g = indexing.getIndex<1>(index - start_delay_data_load);
                    load_data_index.f = indexing.getIndex<2>(index - start_delay_data_load) * BATCH_COMPUTE_FEATURES_SIZE;

                    // switch between registers every iteration
                    bool use_reg_A = (index - start_delay_data_load) % 2 == 0 ? true : false;

                    load_data_reg_A = use_reg_A;
                    // load data

                    pipeline.load_data.address = (ptr4*)(use_reg_A ? &off_A[load_data_index.g][0] : &off_B[load_data_index.g][0]);
                    pipeline.load_data.output = (float4*)(use_reg_A ? d_A[load_data_index.g][load_data_index.f] : d_B[load_data_index.g][load_data_index.f]);

                }

                bool compute_reg_A;
                {
                    // computation is done with double  delay

                    compute_index.s = indexing.getIndex<0>(index - start_delay_compute);
                    compute_index.g = indexing.getIndex<1>(index - start_delay_compute);
                    compute_index.f = indexing.getIndex<2>(index - start_delay_compute) * BATCH_COMPUTE_FEATURES_SIZE;

                    // switch between registers every iteration
                    bool use_reg_A = (index - start_delay_compute) % 2 == 0 ? true : false;

                    compute_reg_A = use_reg_A;
                    // compute
                    pipeline.compute.errors = (float4*)(use_reg_A ? err_vals[compute_index.f] : err_vals[compute_index.f]);
                    pipeline.compute.data = (float4*)(use_reg_A ? d_A[compute_index.g][compute_index.f] : d_B[compute_index.g][compute_index.f]);
                    pipeline.compute.output = &out_val[compute_index.g][s_outer_index * BATCH_MEM_SUBFEATURES_SIZE + compute_index.s][compute_index.f/BATCH_COMPUTE_FEATURES_SIZE];

                }


                // sync only before data buffer is switched
                if (require_sync) {
                    // NOTE: sync is not needed if we have more then enough operations to cover the latency of sore operations
                    // we can rughly say that if there is more then 128 operations then STS latency should be hidden (STS latency should not be more then 100 operations on different platforms)
                    // however since store may be issued half way through operations then use 512 operations as limit
                    if (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y * BATCH_FEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_SIZE * BATCH_MEM_SUBFEATURES_SIZE  < 512) {
						//__syncthreads();
						//__threadfence_block();
#if __CUDA_ARCH__ >= 200
						asm("bar.arrive 15, 1536;"); // # of threads must be greater than 0
#endif

					}
                }

                // pipeline handles loading global (features) data, loading offsets, loading shared (features) data and computing final output

                // pipeline load global -> shared data handles:
                //  - subfeatures:  BATCH_MEM_SUBFEATURES_SIZE
                //
                // pipeline load offset handles:
                //  - subfeatures:  BATCH_MEM_SUBFEATURES_SIZE

                // pipeline compute handles:
                //  - pixels:       BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y
                //  - features:     BATCH_COMPUTE_FEATURES_SIZE
                //  - subfeatures:  one subfeature only
                //  - gauss krn.:   one gaussian kernel only

                // matrix of compute values is of [1,                1,                          BATCH_COMPUTE_FEATURES_SIZE,                       BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y] size
                // computation is repeated        [BATCH_GAUSS_SIZE, BATCH_MEM_SUBFEATURES_SIZE, BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE,   1]-times

                /*if (thread_x == 0 && thread_y == 0 && block_x == 0 && block_y == 0 && f_offset == 0 && s_offset == 0) {
                    if (require_sync)
                        printf("iter: %d, sycned\n", index);

                    printf("pipeline n=%d, s,f=(%d+%d,%d+%d) index %d (g:%d): gl %d (s:%d, f:%d, buff:%d, next buff:%d; reg:%d), off %d (s:%d, f:%d, reg:%d, buff:%d), w %d (s:%d, f:%d, reg:%d), data %d (s:%d, f:%d, buff:%d, next buff:%d; reg:%d), compute %d (s:%d, f:%d, reg:%d)\n",
						    n, s_offset, s_outer_index, f_offset, 0,
                            index, 0,
                            (int)load_global_enabled, load_global.s, load_global.f, global_d, 0, 0,
                            pipeline.load_offset.enabled ? 1 : 0 , load_offset_index.s, load_offset_index.f, (int)load_offset_reg_A, shared_d_next,
                            0, 0, 0, 0,
                            pipeline.load_data.enabled ? 1 : 0 , load_data_index.s, load_data_index.f, shared_d_current, -1, (int)load_data_reg_A,
                            pipeline.compute.enabled ? 1 : 0 , compute_index.s, compute_index.f, (int)compute_reg_A);


                }*/
                //if (f_offset == 12 && s_offset == 4)
                pipeline.execute_step();


#ifdef ENABLE_SHARED_MEM_FOR_OFFSETS
                // if next iteration is not the last one then load offsets and weights for the next one - using double buffering so we do not intereput computation of the current one

// TODO: is this correct or is it a bug ?? !! due to index + 4 ==  NUM_ITER + EXTRA_LOOPS it will not work when NUM_ITER < 2
                if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index + 4 ==  NUM_ITER + EXTRA_LOOPS )

                //if (index + 4 ==  NUM_ITER + EXTRA_LOOPS )

                //if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == 0 && 0)
                // NOTE: we do not need to check for the last index since data should be padded with one extra image to prevent reading invalid data
                if (index == 0)
                {

                    { // TODO: split load_global into two function one for load and one for read to force distance between LDG and STS commands

//                        offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,false,0>(reinterpret_cast<const int4*>(filter_offset_next),
//                                                                                               reinterpret_cast<int4*>(offsets_sh_class.getDataThreadIndexingWrite((s_outer_index + 1 ) % DOUBLE_BUFFERING_OFF)));
                        /*__syncthreads();
                        if (thread_x == 0 && thread_y == 0 && block_x == 0 && block_y == 0 && f_offset == 12 && s_offset == 4) {
                            offsets_sh_class.print();
                        }
                        __syncthreads();*/
                    }

                }
#endif
            }
        }

        /*_image_global_current = _image_global_next;
        _image_global_next = _image_global_current + image_offset;

        _error_global_current = _error_global_current + error_offset;*/
    }

    __syncthreads();

    // we sum results over all threads in this block that worked on the same features into a single value using warp reduce
    typedef cub::WarpReduce<float> WarpReduce;

    // Allocate WarpReduce shared memory for one warp
    __shared__ typename WarpReduce::TempStorage warp_reduce_storage[BlockIndexingT::NUM_WARPS];

    int warp_id = block_indexing.getWarpId();
    WarpReduce warp_reduce(warp_reduce_storage[warp_id]);

#pragma unroll
    for (int g = 0; g < BATCH_GAUSS_SIZE; ++g) {
#pragma unroll
        for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES; ++s) {
#pragma unroll
            for (int f = 0; f < BATCH_FEATURES_SIZE / BATCH_COMPUTE_FEATURES_SIZE; ++f) {
                float fff = out_val[g][s][f].x;
                out_val[g][s][f].x = warp_reduce.Sum(out_val[g][s][f].x);
                out_val[g][s][f].y = warp_reduce.Sum(out_val[g][s][f].y);
                out_val[g][s][f].z = warp_reduce.Sum(out_val[g][s][f].z);
                out_val[g][s][f].w = warp_reduce.Sum(out_val[g][s][f].w);

                //printf("thread (%d, %d) val: %f with sum: %f\n", block_indexing.getThreadId(), warp_id, fff, out_val[g][s][f].x);
            }
        }
    }

    // now we finally write values to global mem
    // only one thread at [0,0] needs to write it (other threads should have the same value)

    if (block_indexing.warp_lane_id() == 0) {
        // TODO: currently supports only Bx*By == WARP_SIZE !! add atomicAdd when Bx*By > WARP_SIZE
        // TODO: we could store float4 at once if output is rearranged indexes to have F as last index

#pragma unroll
        for (int g = 0; g < BATCH_GAUSS_SIZE; ++g) {
#pragma unroll
            for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES ; ++s) {
#pragma unroll
                for (int f = 0; f < BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE; ++f) {
                    atomicAdd(&(output_batch[OFFSET(0, g, s_offset + s, (f_offset/4 + f), 1, G, S, F/4)].x), out_val[g][s][f].x);
                    atomicAdd(&(output_batch[OFFSET(0, g, s_offset + s, (f_offset/4 + f), 1, G, S, F/4)].y), out_val[g][s][f].y);
                    atomicAdd(&(output_batch[OFFSET(0, g, s_offset + s, (f_offset/4 + f), 1, G, S, F/4)].z), out_val[g][s][f].z);
                    atomicAdd(&(output_batch[OFFSET(0, g, s_offset + s, (f_offset/4 + f), 1, G, S, F/4)].w), out_val[g][s][f].w);

                    //if (thread_x == 0 && thread_y == 0 && block_x == 0 && block_y == 0)
                    //    printf("written by thread (%d, %d) at s (%d + %d) f (%d + %d) val: %f\n", block_indexing.getThreadId(), warp_id, s_offset, s, f_offset , f, out_val[g][s][f].x);
                }
            }
        }
    }

#endif
}


template <int Bx, int By, int BATCH_PIXELS_SIZE>
__global__ void
fast_gauss_backward_basic_kernel(const float* filtered_images, const float* error_images,
                                 const int* filter_offsets_x, const int* filter_offsets_y, float* output,
                                 const int I, const int S, const int F, const int G,
                                 const int img_width, const int img_height) {

// INPUT: filtered images  	[I x S x H x W]
//        error images  	[I x S x H x W]
//		  filter offsets   	[F x S x G]
// OUTPUT output  		 	[F x S x G]

#ifndef CUBIN_EMBEDDING

    int thread_x = BATCH_PIXELS_SIZE * (blockIdx.x * Bx + threadIdx.x);
    int thread_y = blockIdx.y * By + threadIdx.y;
    int thread_f = blockIdx.z * 1;

    for (int f = 0; f < 1; ++f) {
        for (int s = 0; s < S; ++s) {
            for (int g = 0; g < G; ++g) {

                float out_value = 0;

                int offset_x = filter_offsets_x[OFFSET(0,s,g,f + thread_f,1,S,G,F)];
                int offset_y = filter_offsets_y[OFFSET(0,s,g,f + thread_f,1,S,G,F)];

                for (int i = 0; i < I; ++i) {
                    float4 x_value, error_value;

                    x_value.x = IS_VALID_PIXEL(thread_x + 0 + offset_x, thread_y + offset_y, img_width, img_height) ? filtered_images[OFFSET(i, s, thread_y + offset_y, thread_x + 0 + offset_x, I, S, img_height, img_width)] : 0;
                    x_value.y = IS_VALID_PIXEL(thread_x + 1 + offset_x, thread_y + offset_y, img_width, img_height) ? filtered_images[OFFSET(i, s, thread_y + offset_y, thread_x + 1 + offset_x, I, S, img_height, img_width)] : 0;
                    x_value.z = IS_VALID_PIXEL(thread_x + 2 + offset_x, thread_y + offset_y, img_width, img_height) ? filtered_images[OFFSET(i, s, thread_y + offset_y, thread_x + 2 + offset_x, I, S, img_height, img_width)] : 0;
                    x_value.w = IS_VALID_PIXEL(thread_x + 3 + offset_x, thread_y + offset_y, img_width, img_height) ? filtered_images[OFFSET(i, s, thread_y + offset_y, thread_x + 3 + offset_x, I, S, img_height, img_width)] : 0;

                    error_value.x = IS_VALID_PIXEL(thread_x + 0 + offset_x, thread_y + offset_y, img_width, img_height) ? error_images[OFFSET(i, s, thread_y, thread_x + 0, I, S, img_height, img_width)] : 0;
                    error_value.y = IS_VALID_PIXEL(thread_x + 1 + offset_x, thread_y + offset_y, img_width, img_height) ? error_images[OFFSET(i, s, thread_y, thread_x + 1, I, S, img_height, img_width)] : 0;
                    error_value.z = IS_VALID_PIXEL(thread_x + 2 + offset_x, thread_y + offset_y, img_width, img_height) ? error_images[OFFSET(i, s, thread_y, thread_x + 2, I, S, img_height, img_width)] : 0;
                    error_value.w = IS_VALID_PIXEL(thread_x + 3 + offset_x, thread_y + offset_y, img_width, img_height) ? error_images[OFFSET(i, s, thread_y, thread_x + 3, I, S, img_height, img_width)] : 0;

                    out_value += x_value.x * error_value.x +
                                            x_value.y * error_value.y +
                                            x_value.z * error_value.z +
                                            x_value.w * error_value.w;

                }

                atomicAdd(output + OFFSET(0, s, g, f + thread_f, 1, S, G, F), out_value);

                //printf("atomicAdd to loc %d of value %f, final out value was %f\n", OFFSET(0,s,g,f,1,S,G,F), out_value, output[OFFSET(0,s,g,f,1,S,G,F)]);
            }
        }
    }

#endif
}

#include <iostream>

#define OFFSET8(i8, i7, i6, i5, i4, i3, i2, i1, num_i8, num_i7, num_i6, num_i5, num_i4, num_i3, num_i2, num_i1) \
				((( (( ( ((i8) * (num_i7) + i7)* (num_i6)  + (i6)  )*(num_i5)  + (i5)  )   * (num_i4) + (i4))*(num_i3) + (i3)) * (num_i2) + (i2))*(num_i1) + (i1) )

template <typename BlockIndexingT>
__global__  void
perpare_weights_and_offsets_bw(const float* filter_weights, const int* filter_offsets_x, const int* filter_offsets_y, float *prepared_filter_weights, int *prepared_filter_offsets, int S, int G, int F) {

	static const int NUM_SM = BlockIndexingT::NUM_SM;
	static const int Bx = BlockIndexingT::Bx;
	static const int By = BlockIndexingT::By;
	static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
    static const int BLOCK_SUBFEATURES = BlockIndexingT::BLOCK_SUBFEATURES;
	static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
	static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
	static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
	static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
	static const int BATCH_GAUSS_SIZE = BlockIndexingT::BATCH_GAUSS_SIZE;
	static const int IMG_WIDTH = BlockIndexingT::IMG_WIDTH;
	static const int IMG_HEIGHT = BlockIndexingT::IMG_HEIGHT;
	static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;
	static const int NUM_THREADS = BlockIndexingT::NUM_THREADS;

	// inputs in quad vectors
	const float4* filter_weights4 = reinterpret_cast<const float4*>(filter_weights);
	const int4* filter_offsets_x4 = reinterpret_cast<const int4*>(filter_offsets_x);
	const int4* filter_offsets_y4 = reinterpret_cast<const int4*>(filter_offsets_y);

	// outputs in quad vectors
	float4* prepared_filter_weights4 = reinterpret_cast<float4*>(prepared_filter_weights);
	int4* prepared_filter_offsets4 = reinterpret_cast<int4*>(prepared_filter_offsets);


	int f_input_index = blockIdx.x * blockDim.x  + threadIdx.x;
	int g_input_index = blockIdx.y * blockDim.y  + threadIdx.y;
	int s_input_index = blockIdx.z * blockDim.z  + threadIdx.z;

	// input data is of the form
	// float4 of size [S x G x F]

	int input_index = (( s_input_index )*G + g_input_index ) * F + f_input_index;

	// output data is of the form:
	// float4 of size [F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES)] x [S / (BATCH_MEM_SUBFEATURES_SIZE*BLOCK_SUBFEATURES)] x [G / BATCH_GAUSS_SIZE]
	//				 	x [BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES] x [BATCH_GAUSS_SIZE] x [PIXELS_INTERPOLATION_SIZE] x [BATCH_FEATURES_SIZE/4] x [BLOCK_FEATURES];

	static const int dim1_size = BLOCK_FEATURES;
	static const int dim2_size = BATCH_FEATURES_SIZE/4;
	static const int dim3_size = 4;
	static const int dim4_size = BATCH_GAUSS_SIZE;
	static const int dim5_size = BATCH_MEM_SUBFEATURES_SIZE * BLOCK_SUBFEATURES;

	int dim6_size = G / dim4_size;
	int dim7_size = S / dim5_size;
	int dim8_size = F / (dim1_size * dim2_size);


	int main_f_index = f_input_index / (dim1_size * dim2_size);
	int f_block_tid = f_input_index % (dim1_size * dim2_size);

	int main_s_index = s_input_index / dim5_size;
	int s_block_tid = s_input_index % dim5_size;

	int main_g_index = g_input_index / dim4_size;
	int g_block_tid = g_input_index % dim4_size;

	int s_mem_index = s_block_tid;
	int g_index = g_block_tid;

	// switch between block and batch indexes so that consecutive features and stored in [BATCH_FEATURES_SIZE/4]
	int f_batch_index = f_block_tid % (dim2_size);
	int f_block_index = f_block_tid / (dim2_size);


	int output_index = OFFSET8(main_f_index,
								main_s_index,
								main_g_index,
								s_mem_index,
								g_index,
								0,
								f_batch_index,
								f_block_index,
								dim8_size, dim7_size, dim6_size, dim5_size, dim4_size, 1, dim2_size, dim1_size);

	/*printf("input index %d goes to output index %d: input s: %d, g: %d, f: %d: output dims: %d, %d, %d, %d, %d, %d, %d\n", input_index, output_index, s_input_index, g_input_index, f_input_index,main_f_index,
			main_s_index,
			main_g_index,
			s_mem_index,
			f_batch_index,
			g_index,
			f_block_index);
*/

	// for offsets we need to combine X and Y coordinates and transform them directly to int values approproate for using specific BLOCK_ and BATCH_ sizes

	int4 offset_x = filter_offsets_x4[input_index];
	int4 offset_y = filter_offsets_y4[input_index];

	// offset is relative to shared memory organization which is defined by  BlockSharedMemory parameters:
	//		- SharedMem::ALLOC_WIDTH
	//		- SharedMem::ALLOC_HEIGHT

	static const int BATCH_COMPUTE_FEATURES_SIZE = 4;

	// using float4 to load so use
	static const int BATCH_SH_PIXELS_SIZE = 4;

	static const int DOUBLE_BUFFERING = 2;

	typedef BlockSharedMemory<NUM_THREADS,
								Bx * BATCH_PIXELS_SIZE_X,
								By * BATCH_PIXELS_SIZE_Y,
								MAX_OFFSET,
								DOUBLE_BUFFERING * BATCH_MEM_SUBFEATURES_SIZE,
								float4,
								BATCH_SH_PIXELS_SIZE> SharedMem;

	int4 output_offset;

	//printf("offset at index %d, s: %d, g: %d, f: %d, has been transform from %d,%d to %d\n", input_index, output_index, s_input_index, g_input_index, f_input_index, offset_y.x, offset_y.y, output_offset.x);

	// offset should be in bytes !!! (not in 4 bytes as for float or 16 bytes as for float4)
	output_offset.x = (offset_y.x * (SharedMem::PITCHED_WIDTH) + offset_x.x) * sizeof(float);
	output_offset.y = (offset_y.x * (SharedMem::PITCHED_WIDTH) + offset_x.y) * sizeof(float);
	output_offset.z = (offset_y.z * (SharedMem::PITCHED_WIDTH) + offset_x.z) * sizeof(float);
	output_offset.w = (offset_y.w * (SharedMem::PITCHED_WIDTH) + offset_x.w) * sizeof(float);

	// If offset is to odd number then access to shared memory will not be alligned (e.g. cannot use float2)
	// so we replicate data in shared memory for accesses to odd offsets with addintional buffer.
	// We now need to ensure that offset here will address that buffer instead - buffers are stored one after
	// another so we just need to add the size of one buffer minus the alignment value in x dimension

	// TODO: check if we are still replicating image and disable this part if not!!!
	if (output_offset.x % 2 == 1) output_offset.x += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
	if (output_offset.y % 2 == 1) output_offset.y += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
	if (output_offset.z % 2 == 1) output_offset.z += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
	if (output_offset.w % 2 == 1) output_offset.w += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;

	prepared_filter_offsets4[output_index] = output_offset;
/*
	// for weights we integrate interpolation values into four sets of weights

	int output_index_0 =  OFFSET8(main_f_index,
									main_s_index,
									main_g_index,
									s_mem_index,
									g_index,
									0,
									f_batch_index,
									f_block_index,
									dim8_size, dim7_size, dim6_size, dim5_size, dim4_size, dim3_size, dim2_size, dim1_size);

	// prepare factors for interpolation

	float4 interp_offset_y,interp_offset_x;

	interp_offset_x.x = offset_x.x - (float)(int)(offset_x.x);
	interp_offset_x.y = offset_x.y - (float)(int)(offset_x.y);
	interp_offset_x.z = offset_x.z - (float)(int)(offset_x.z);
	interp_offset_x.w = offset_x.w - (float)(int)(offset_x.w);

	float4 factor_00, factor_01, factor_10, factor_11;

	factor_11.x = interp_offset_x.x * interp_offset_y.x;
	factor_11.y = interp_offset_x.y * interp_offset_y.y;
	factor_11.z = interp_offset_x.z * interp_offset_y.z;
	factor_11.w = interp_offset_x.w * interp_offset_y.w;

	factor_10.x = (interp_offset_x.x) * (1-interp_offset_y.x);
	factor_10.y = (interp_offset_x.y) * (1-interp_offset_y.y);
	factor_10.z = (interp_offset_x.z) * (1-interp_offset_y.z);
	factor_10.w = (interp_offset_x.w) * (1-interp_offset_y.w);

	factor_01.x = (1-interp_offset_x.x) * (interp_offset_y.x);
	factor_01.y = (1-interp_offset_x.y) * (interp_offset_y.y);
	factor_01.z = (1-interp_offset_x.z) * (interp_offset_y.z);
	factor_01.w = (1-interp_offset_x.w) * (interp_offset_y.w);

	factor_00.x = (1-interp_offset_x.x) * (1-interp_offset_y.x);
	factor_00.y = (1-interp_offset_x.y) * (1-interp_offset_y.y);
	factor_00.z = (1-interp_offset_x.z) * (1-interp_offset_y.z);
	factor_00.w = (1-interp_offset_x.w) * (1-interp_offset_y.w);

	// create weights with interpolation factors
	prepared_filter_weights4[output_index_0].x = filter_weights4[input_index].x * factor_00.x;
	prepared_filter_weights4[output_index_0].y = filter_weights4[input_index].y * factor_00.y;
	prepared_filter_weights4[output_index_0].z = filter_weights4[input_index].z * factor_00.z;
	prepared_filter_weights4[output_index_0].w = filter_weights4[input_index].w * factor_00.w;

	int output_index_1 = output_index_0 + 1 *  (dim1_size * dim2_size);
	prepared_filter_weights4[output_index_1].x = filter_weights4[input_index].x * factor_01.x;
	prepared_filter_weights4[output_index_1].y = filter_weights4[input_index].y * factor_01.y;
	prepared_filter_weights4[output_index_1].z = filter_weights4[input_index].z * factor_01.z;
	prepared_filter_weights4[output_index_1].w = filter_weights4[input_index].w * factor_01.w;

	int output_index_2 = output_index_0 + 2 *  (dim1_size * dim2_size);
	prepared_filter_weights4[output_index_2].x = filter_weights4[input_index].x * factor_10.x;
	prepared_filter_weights4[output_index_2].y = filter_weights4[input_index].y * factor_10.y;
	prepared_filter_weights4[output_index_2].z = filter_weights4[input_index].z * factor_10.z;
	prepared_filter_weights4[output_index_2].w = filter_weights4[input_index].w * factor_10.w;

	int output_index_3 = output_index_0 + 3 *  (dim1_size * dim2_size);
	prepared_filter_weights4[output_index_3].x = filter_weights4[input_index].x * factor_11.x;
	prepared_filter_weights4[output_index_3].y = filter_weights4[input_index].y * factor_11.y;
	prepared_filter_weights4[output_index_3].z = filter_weights4[input_index].z * factor_11.z;
	prepared_filter_weights4[output_index_3].w = filter_weights4[input_index].w * factor_11.w;
*/
}


__global__ void
print2d_array_bw(const float* input, int width, int height, int N) {

	for (int n = 0; n < N; ++n)  {
		printf("input image %d: \n",n);
		for (int j = 0; j < height; ++j)  {
			for (int i = 0; i < width; ++i) {
				printf("%f ",input[( n* height + j ) * width + i]);
			}
			printf("\n");
		}
		printf("\n");
	}

}


template <>
void fast_gauss_backward<double>(const double* filtered_images, const double* error_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
								const double* filter_weights, double* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height, cudaStream_t streamId) {

}


float* create_input_with_border_bw(const float* filtered_images, const int img_width, const int img_height, const int NN, const int border_size, cudaStream_t* streams, const int NUM_STREAMS ) {

	int new_width = img_width + 2*border_size;
	int new_height = img_height + 2*border_size;

	float* filtered_images_with_border;
	cudaMalloc(&filtered_images_with_border, sizeof(float) * new_width * new_height * (NN +1)); // NOTE: we add extra image as padding to prevent double buffering from load invalid data on last batch

	//cudaMemset(filtered_images_with_border, 0,  sizeof(float) * new_width * new_height * NN);

	caffe_gpu_set<float>(new_width * new_height * NN, 1.0f, filtered_images_with_border);

	for (int n = 0; n < NN; ++n) {
		// position at begining of sub-feature input map
		float* dst = filtered_images_with_border + n*new_width * new_height;
		const float* src = filtered_images + n * img_width * img_height;

		// move destination pointr to begining of actual data i.e. to [border_size,border_size] position
		dst += new_width * border_size + border_size;

		cudaMemcpy2D(dst, sizeof(float) * new_width, src, sizeof(float) * img_width, sizeof(float) * img_width, img_height,  cudaMemcpyDeviceToDevice);

	}

	return filtered_images_with_border;
}

template <>
void fast_gauss_backward<float>(const float* filtered_images, const float* error_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
								const float* filter_weights, float* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height, cudaStream_t streamId) {


	static const int CUDA_THREADS = 256;
	static const int BATCH_PIXELS_SIZE = 4;

	static const int BLOCK_X = 32/BATCH_PIXELS_SIZE;
	static const int BLOCK_Y = 16;

	static const int BATCH_FEATURES_SIZE = 16;
	static const int BATCH_SUBFEATURES_SIZE = 2;
	static const int MAX_OFFSET = 4;

	dim3 threadsPerBlock(BLOCK_X, BLOCK_Y, 1);

	dim3 numBlocks( ((int)ceil(img_width/(float)BATCH_PIXELS_SIZE) + threadsPerBlock.x - 1) / threadsPerBlock.x,	// over image width (N pixels per thread where N=BATCH_PIXELS_SIZE)
					((int)ceil(img_height) + threadsPerBlock.y - 1) / threadsPerBlock.y,							// over image height
                    F/1);																		// over S*G*F

	std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
	std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

	cudaDeviceProp devProp;
	CUDA_CHECK(cudaGetDeviceProperties(&devProp,0));

	std::cout << "CUDA requrements: texture alignment " << devProp.textureAlignment << ", pitch alingnment " << devProp.texturePitchAlignment << std::endl;

    if (0) {

        std::cout << "started fast_gauss_backward_basic_kernel" << std::endl;

        clock_t start_t = clock();
        fast_gauss_backward_basic_kernel<BLOCK_X,BLOCK_Y,BATCH_PIXELS_SIZE><<<numBlocks,threadsPerBlock>>>(filtered_images, error_images, filter_offsets_x, filter_offsets_y, output, I, S, F, G, img_width, img_height);
        std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
        cudaDeviceSynchronize();

        clock_t end_t = clock();
        std::cout << "fast_gauss_backward_basic_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
        //
    }

	if (1) {

		clock_t start_tx_create = clock();

		// prepare cuArray and texture obj for each image
		struct cudaExtent cuArray_size = make_cudaExtent(img_width / 4, img_height, S);
		cudaArray** cuArray_list = new cudaArray*[I];

		// create texture for each image in batch (current HW limit is 256 textures)
		cudaTextureObject_t* filtered_images_tex_layered = new cudaTextureObject_t[I];

		struct cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

		// prepare params for copying
		struct cudaMemcpy3DParms copy_param = {0};

		copy_param.srcPos = make_cudaPos(0,0,0);
		//copy_param.srcPtr;
		//copy_param.dstArray;
		copy_param.extent = cuArray_size;
		copy_param.kind = cudaMemcpyDeviceToDevice;

		// prepare params for texture resource type
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));

		resDesc.resType = cudaResourceTypeArray;
		//resDesc.res.array.array;

		// prepare params for texture addressing and filtering
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0]   = cudaAddressModeBorder;
		texDesc.addressMode[1]   = cudaAddressModeBorder;

		//texDesc.filterMode       = cudaFilterModeLinear;
		texDesc.filterMode       = cudaFilterModePoint;
		texDesc.readMode         = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		// create a list of  2D layered images and assign each feature to a layer in a single image
		for (int i = 0; i < I; ++i) {

			// allocate memory
			CUDA_CHECK(cudaMalloc3DArray(&cuArray_list[i], &channelDesc, cuArray_size, cudaArrayLayered));

			// set resource pointer
			resDesc.res.array.array = cuArray_list[i];

			// create texture
			CUDA_CHECK(cudaCreateTextureObject(&filtered_images_tex_layered[i], &resDesc, &texDesc, NULL));
		}

		clock_t end_tx_create = clock();
		std::cout << "texture created  in " << (((float)(end_tx_create-start_tx_create))/CLOCKS_PER_SEC) << std::endl;

		clock_t start_tx_cpy = clock();
		for (int i = 0; i < I; ++i) {

			// prepare src and dst pointers for copying data
			copy_param.srcPtr = make_cudaPitchedPtr((void*)&filtered_images[i * (S * img_width * img_height)], img_width * sizeof(float), img_width, img_height);
			copy_param.dstArray = cuArray_list[i];

			// copy data
			CUDA_CHECK(cudaMemcpy3D(&copy_param));
		}

		// copy array cudaTextureObject_t references to GPU memory
		cudaTextureObject_t* filtered_images_tex_layered_gpu;

		cudaMalloc(&filtered_images_tex_layered_gpu, sizeof(cudaTextureObject_t) * I);
		cudaMemcpy(filtered_images_tex_layered_gpu, filtered_images_tex_layered, sizeof(cudaTextureObject_t)* I, cudaMemcpyHostToDevice);

		clock_t end_tx_cpy = clock();
		std::cout << "texture copied  in " << (((float)(end_tx_cpy-start_tx_cpy))/CLOCKS_PER_SEC) << std::endl;



		if (1) {

            // each block of multiple threads handles:
            //  - pixel:        BLOCK_X * BLOCK_Y
            //  - features:     BLOCK_FEATURES * BATCH_FEATURES_SIZE
            //  - subfeatures:  BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE
            //  - gauss krn:    BATCH_GAUSS_SIZE

            // within block each thread handles:
            //  - pixels:       BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y
            //  - features:     BATCH_FEATURES_SIZE
            //  - subfeatures:  BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE
            //  - gauss krn:    BATCH_GAUSS_SIZE

            // each thread handles features and subfeatures as:
            //  - features:     one warp always handles only BATCH_FEATURES_SIZE features, but N warps are used for different features where N=BLOCK_FEATURES
            //  - subfeatures:  at once only BATCH_MEM_SUBFEATURES_SIZE subfeatures are loaded, but N-times iterated over that where N=BLOCK_SUBFEATURES

			static const int BATCH_PIXELS_SIZE_X = 2;
			static const int BATCH_PIXELS_SIZE_Y = 8;

			static const int PIXELS_INTERPOLATION_Dx = 1;
			static const int PIXELS_INTERPOLATION_Dy = 1;

			static const int BLOCK_X = 32/BATCH_PIXELS_SIZE_X;
			static const int BLOCK_Y = 16/BATCH_PIXELS_SIZE_Y;
			/* best with global loading
			static const int BLOCK_FEATURES = 4;
            static const int BLOCK_SUBFEATURES = 4;

			static const int BATCH_FEATURES_SIZE = 8;
			static const int BATCH_MEM_SUBFEATURES_SIZE = 4;
			static const int BATCH_GAUSS_SIZE = 1;*/
            static const int BLOCK_FEATURES = 8;
            static const int BLOCK_SUBFEATURES = 4;

            static const int BATCH_FEATURES_SIZE = 8;
            static const int BATCH_MEM_SUBFEATURES_SIZE = 1;
            static const int BATCH_GAUSS_SIZE = 1;
			/* best wih no global loading !!
            static const int BLOCK_FEATURES = 8;
            static const int BLOCK_SUBFEATURES = 8;

            static const int BATCH_FEATURES_SIZE = 4;
            static const int BATCH_MEM_SUBFEATURES_SIZE = 4;
            static const int BATCH_GAUSS_SIZE = 1; //*/
			static const int IMG_WIDTH = 64;
			static const int IMG_HEIGHT = 64;
			static const int MAX_OFFSET = 4;

			static const int NUM_SM = 1; // number of streaming multiprocessors

			typedef class BlockIndexing<NUM_SM,
							BLOCK_X, BLOCK_Y, BLOCK_FEATURES, BLOCK_SUBFEATURES,
							BATCH_PIXELS_SIZE_X, BATCH_PIXELS_SIZE_Y,
							PIXELS_INTERPOLATION_Dx, PIXELS_INTERPOLATION_Dy,
							BATCH_FEATURES_SIZE,
							BATCH_MEM_SUBFEATURES_SIZE,
							BATCH_GAUSS_SIZE,
							IMG_WIDTH, IMG_HEIGHT,
							MAX_OFFSET> BlockIndexingPipelineT;

			BlockIndexingPipelineT::Launch block_indexing;

			dim3 threadsPerBlock = block_indexing.getThreadsPerBlock(I, F, S, img_width, img_height);

			dim3 numBlocks = block_indexing.getBlocksPerGrid(I, F, S, img_width, img_height);

			float* filtered_images_with_border;
			{
				std::cout << "started create_input_with_border_bw" << std::endl;

				const int NUM_STREAMS = S;

				cudaStream_t* streams = new cudaStream_t[NUM_STREAMS];

				for (int i = 0; i < NUM_STREAMS; ++i) {
					cudaStreamCreate(&streams[i]);
				}

				clock_t start_t = clock();
				filtered_images_with_border = create_input_with_border_bw(filtered_images, img_width, img_height, I*S, MAX_OFFSET, streams, NUM_STREAMS);
				CUDA_POST_KERNEL_CHECK;
				std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
				cudaDeviceSynchronize();

				clock_t end_t = clock();
				std::cout << "create_input_with_border_bw in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

				for (int i = 0; i < NUM_STREAMS; ++i) {
					cudaStreamDestroy(streams[i]);
				}

			}

			float* prepared_filter_weights;
			int* prepared_filter_offsets;

			{

				// allocate additional block of data so that double buffering can load valid data on last batch
				static const int OFFSET_BLOCK_MEM_SIZE = BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE *  BLOCK_FEATURES;

                //static const int WEIGHT_BLOCK_MEM_SIZE = BLOCK_SUBFEATURES * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_Dx*PIXELS_INTERPOLATION_Dy * BATCH_FEATURES_SIZE * BLOCK_FEATURES;
				//cudaMalloc(&prepared_filter_weights, sizeof(float) * ( 4*S*G*F + WEIGHT_BLOCK_MEM_SIZE));

				cudaMalloc(&prepared_filter_offsets, sizeof(float) * ( S*G*F + OFFSET_BLOCK_MEM_SIZE));
                cudaMemset(prepared_filter_offsets,0, sizeof(float) * ( S*G*F + OFFSET_BLOCK_MEM_SIZE));

				//dim3 threadsPerBlock(16, 1, 16);
                dim3 threadsPerBlock(4, 1, 4);
				dim3 numBlocks((int)ceil((F/4)/threadsPerBlock.x),
								(int)ceil(G/threadsPerBlock.y),
								(int)ceil(S/threadsPerBlock.z));

				std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
				std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

				std::cout << "started copy_permute_weights" << std::endl;

				clock_t start_t = clock();
// TODO!!!		perpare_weights_and_offsets_bw<BlockIndexingPipelineT><<<numBlocks,threadsPerBlock>>>(filter_weights, filter_offsets_x, filter_offsets_y, prepared_filter_weights, prepared_filter_offsets, S, G, F/4);


				CUDA_POST_KERNEL_CHECK;
				std::cout << "waiting for copy_permute_weights" << std::endl;
				cudaDeviceSynchronize();

				clock_t end_t = clock();
				std::cout << "copy_permute_weights in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
                /*
				float* prepared_filter_offsets_cpu = new float[S*F*G];

				for (int i = 0; i < S*G*F; ++i)
					prepared_filter_offsets_cpu[i] = -1;

				cudaMemcpy(prepared_filter_offsets_cpu, prepared_filter_weights, sizeof(int)* S*F*G, cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();

				for (int i = 0; i < S*G*F; ++i) {
					std::cout << prepared_filter_offsets_cpu[i ] << " ";
				}
				std::cout << std::endl;
				return;*/
			}

			std::cout << "started fast_gauss_backward_pipeline_kernel" << std::endl;

			std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
			std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;


			clock_t start_t = clock();
			fast_gauss_backward_pipeline_kernel<BlockIndexingPipelineT><<<numBlocks,threadsPerBlock>>>(filtered_images_tex_layered_gpu, filtered_images_with_border, error_images, prepared_filter_offsets, prepared_filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
			CUDA_POST_KERNEL_CHECK;
			std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
			cudaDeviceSynchronize();

			clock_t end_t = clock();
			std::cout << "fast_gauss_backward_pipeline_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

			cudaFree(prepared_filter_weights);
			cudaFree(prepared_filter_offsets);

			cudaFree(filtered_images_with_border);
		}

		for (int i = 0; i < I; ++i) {
			cudaFreeArray(cuArray_list[i]);
		}

		cudaFree(filtered_images_tex_layered_gpu);

		free(cuArray_list);
		free(filtered_images_tex_layered);
	}


}


}  // namespace caffe


