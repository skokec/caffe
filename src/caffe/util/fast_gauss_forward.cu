#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <cmath>

#include "glog/logging.h"

#include "caffe/util/fast_gauss_forward.hpp"
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
			int _BATCH_PIXELS_SIZE_X,
			int _BATCH_PIXELS_SIZE_Y,
			int _PIXELS_INTERPOLATION_Dx,
			int _PIXELS_INTERPOLATION_Dy,
			int _BATCH_FEATURES_SIZE,
			int _BATCH_COMPUTE_SUBFEATURES_SIZE,
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
		BATCH_PIXELS_SIZE_X = _BATCH_PIXELS_SIZE_X,
		BATCH_PIXELS_SIZE_Y = _BATCH_PIXELS_SIZE_Y,
		PIXELS_INTERPOLATION_Dx = _PIXELS_INTERPOLATION_Dx,
		PIXELS_INTERPOLATION_Dy = _PIXELS_INTERPOLATION_Dy,
		BATCH_FEATURES_SIZE = _BATCH_FEATURES_SIZE,
		BATCH_COMPUTE_SUBFEATURES_SIZE = _BATCH_COMPUTE_SUBFEATURES_SIZE,
		BATCH_MEM_SUBFEATURES_SIZE = _BATCH_MEM_SUBFEATURES_SIZE,
		BATCH_GAUSS_SIZE = _BATCH_GAUSS_SIZE,
		IMG_WIDTH = _IMG_WIDTH,
		IMG_HEIGHT = _IMG_HEIGHT,
		MAX_OFFSET = _MAX_OFFSET,
		NUM_THREADS = Bx* By * BLOCK_FEATURES
	};

	// CPU only functions
	class Launch {
	public:
		dim3 getThreadsPerBlock(int num_images, int num_features, int num_subfeatures, int img_width, int img_height) {
			// number of threads per blocks
			return dim3(Bx * By * BLOCK_FEATURES, 1, 1);
		}

		dim3 getBlocksPerGrid(int num_images, int num_features, int num_subfeatures, int img_width, int img_height) {

			// number of blocks per kernel launch
			return dim3 ( NUM_SM * (int)ceil(num_features/(BLOCK_FEATURES * BATCH_FEATURES_SIZE)),
						(int)ceil(img_width /  (float)(Bx * BATCH_PIXELS_SIZE_X) ) * (int)ceil(img_height / (float)(By * BATCH_PIXELS_SIZE_Y) ),
						(int)ceil(num_images/NUM_SM)
						);
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

		__device__ Kernel(int img_width, int img_height) {
			img_size.x = img_width;
			img_size.y = img_height;

			f_thread_idx = threadIdx.x / (Bx * By);
			px_thread_idx = threadIdx.x % (Bx * By);

			img_block_idx =  blockIdx.x % NUM_SM;
			f_block_idx =  blockIdx.x / NUM_SM;
		}

		// return global image index that specific thread handles
		__device__ int getImageIdx() {
			return blockIdx.z * NUM_SM  + img_block_idx;
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
			return 0;
		}

		__device__ int2 getPosBlockSize() {
			return make_int2(Bx * BATCH_PIXELS_SIZE_X,
							 By * BATCH_PIXELS_SIZE_Y);
		}

		__device__ int2 getPosBlockIdx() {

			int blockIdx_x = blockIdx.y % (img_size.x / (Bx * BATCH_PIXELS_SIZE_X));
			int blockIdx_y = blockIdx.y / (img_size.x / (Bx * BATCH_PIXELS_SIZE_X));

			return make_int2(BATCH_PIXELS_SIZE_X * (blockIdx_x * Bx),
							 BATCH_PIXELS_SIZE_Y * (blockIdx_y * By));
		}

		__device__ int2 getPosThreadIdx() {

			int threadIdx_x = px_thread_idx % (Bx);
			int threadIdx_y = px_thread_idx / (Bx);

			return make_int2(BATCH_PIXELS_SIZE_X * threadIdx_x,
							 BATCH_PIXELS_SIZE_Y * threadIdx_y);
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

	float* storage_data_for_writing;
	float* storage_data_for_reading;


	_Data& storage;

	// thread indexing for storing/writing data from global mem
	int2 thread_indexing_writing;

	// thread indexing for reading data by each thread (MUST be user defined in constructor)
	int2 thread_indexing_reading;
public:

	typedef _Data Data;

	__device__
	BlockSharedMemory(Data &_storage, int2 read_thread_idx) : storage(_storage), thread_indexing_reading(read_thread_idx) {
		thread_indexing_writing = calcThreadIdx();
		storage_data_for_writing = getDataAt(0, thread_indexing_writing.x/ BATCH_ELEMENTS, thread_indexing_writing.y);
		storage_data_for_reading = getDataAt(0, (thread_indexing_reading.x + APRON_SIZE) / BATCH_ELEMENTS, thread_indexing_reading.y + APRON_SIZE);
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
	void load_global(const ELEMENT_TYPE* global_data, ELEMENT_TYPE* shared_data, int GLOBAL_DATA_WIDTH = -1) {

		if (GLOBAL_DATA_WIDTH < 0)
			GLOBAL_DATA_WIDTH = _GLOBAL_DATA_WIDTH;

		// global_data MUST be positioned at [0,0] in global data without APRON, i.e., at [APRON,APRON] / BATCH_ELEMENTS  in shared storage.data
//printf("started global read\n");
		#pragma unroll
		for (int j = -APRON_SIZE; j < HEIGHT + APRON_SIZE; j+=NUM_THREADS_HEIGHT) {
			#pragma unroll
			for (int i = -APRON_SIZE; i < WIDTH + APRON_SIZE; i+=NUM_THREADS_WIDTH * BATCH_ELEMENTS) {
				// current_image already at position for this block

				float4 tmp;
				// USING TEXTURE - working
				//tmp = tex2DLayered<float4>(load_global.texture, (block_x + thread_indexing.x + i) / BATCH_SH_PIXELS_SIZE + 0.5f, block_y + j + thread_indexing.y + 0.5f, load_global.texture_layer);

				if (thread_indexing_writing.x < (WIDTH + APRON_SIZE - i)  && thread_indexing_writing.y < HEIGHT + APRON_SIZE - j)  {

					ELEMENT_TYPE tmp;

					// USING GLOBAL - working
					if (USE_FILL) {
						tmp.x = FILL_VALUE; tmp.y = FILL_VALUE; tmp.z = FILL_VALUE; tmp.w = FILL_VALUE;
					} else {
						tmp = global_data[j * GLOBAL_DATA_WIDTH / BATCH_ELEMENTS + i / BATCH_ELEMENTS];
//						printf("loading global data: %d,%d, values %f, %f, %f, %f\n", i + thread_indexing_writing.x, j+thread_indexing_writing.y, tmp.x, tmp.y, tmp.z, tmp.w);
					}

					int write_offset = (j + APRON_SIZE) * ALLOC_WIDTH  + (i + APRON_SIZE) / BATCH_ELEMENTS;

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
//printf("end global read\n");
//		__syncthreads();

//		print();
//		__syncthreads();
	}


	__device__
	void print() {
		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.y == 0) {
			__syncthreads();

			printf("printing shared memory:\n");

			for (int s = 0; s < NUM_BUFFER_REPEAT; ++s) {
				for (int j = 0; j < ALLOC_HEIGHT; ++j){
					for (int i = 0; i < ALLOC_WIDTH; ++i){
						float4 tmp = storage.data[s][j][i];
						printf("%f %f %f %f ", tmp.x, tmp.y, tmp.z, tmp.w);
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
		int BATCH_COMPUTE_SUBFEATURES_SIZE,
		int BATCH_MEM_SUBFEATURES_SIZE,
		int BLOCK_FEATURES,
		int IMG_WIDTH, int IMG_HEIGHT,
		typename  _BlockSharedMemoryT>
class PipelineEngine {

	enum {
		PIXELS_INTERPOLATION_SIZE = PIXELS_INTERPOLATION_Dx * PIXELS_INTERPOLATION_Dy
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

	// load w
	struct {
		bool enabled;
		float4* address;
		float4* output;
	} load_weights;

	// load data
	struct {
		bool enabled;
		ptr4* address;
		float4* output;	// [BATCH_F][BATCH_PIXELS/4]
	} load_data;

	// compute
	struct {
		bool enabled;
		float4* weights;
		float4* data;
		float4* output; // [BATCH_F][BATCH_PIXELS/4]
	} compute;

	// block
	int block_x;
	int block_y;


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

		// load quad of w for next one
		if (load_weights.enabled) {
			// p.next_w = *(float4)p.next_w_address
			for (int i = 0; i < PIXELS_INTERPOLATION_SIZE; ++i) {
				for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/4; ++f_quad_index ) {
					COPY_VECTOR4(load_weights.output[i* BATCH_COMPUTE_FEATURES_SIZE/4 + f_quad_index],load_weights.address[(i * BATCH_FEATURES_SIZE/4  + f_quad_index) * BLOCK_FEATURES]); // weights for F[0], F[1], F[2], F[3]

//					if (threadIdx.x == 0 && blockIdx.x == 0) {
//						printf("loaded weights: %f, %f, %f, %f\n", load_weights.output[i + f_quad_index].x, load_weights.output[i + f_quad_index].y, load_weights.output[i + f_quad_index].z, load_weights.output[i + f_quad_index].w);
//					}
				}
			}
		}

		// load quad of offsets for next one and make it directly into pointer to data
		if (load_offset.enabled) {
			//*(p.next_offset) = *((float4*)p.next_offset_address);

			for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/4; ++f_quad_index ) {
				load_offset.output[f_quad_index].quad[0] = (float*)((void*)load_offset.base_address + load_offset.offset_address[f_quad_index].x); // F[0]
				load_offset.output[f_quad_index].quad[1] = (float*)((void*)load_offset.base_address + load_offset.offset_address[f_quad_index].y); // F[1]
				load_offset.output[f_quad_index].quad[2] = (float*)((void*)load_offset.base_address + load_offset.offset_address[f_quad_index].z); // F[2]
				load_offset.output[f_quad_index].quad[3] = (float*)((void*)load_offset.base_address + load_offset.offset_address[f_quad_index].w); // F[3]
/*
				if (load_offset.offset_address[f_quad_index].x != 0) printf("found invalid offset value %d, at f_quad_index: %d, [%d]\n",load_offset.offset_address[f_quad_index].x, f_quad_index, 0);
				if (load_offset.offset_address[f_quad_index].y != 0) printf("found invalid offset value %d, at f_quad_index: %d, [%d]\n",load_offset.offset_address[f_quad_index].x, f_quad_index, 1);
				if (load_offset.offset_address[f_quad_index].z != 0) printf("found invalid offset value %d, at f_quad_index: %d, [%d]\n",load_offset.offset_address[f_quad_index].x, f_quad_index, 2);
				if (load_offset.offset_address[f_quad_index].w != 0) printf("found invalid offset value %d, at f_quad_index: %d, [%d]\n",load_offset.offset_address[f_quad_index].x, f_quad_index, 3);
*/
			}
		}

		NDIndexing<BATCH_COMPUTE_FEATURES_SIZE,
					NDIndexing<PIXELS_INTERPOLATION_Dy,
						NDIndexing<PIXELS_INTERPOLATION_Dx,
							NDIndexingZero<(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4> > > > indexing;

		#pragma unroll
		for (int i = 0; i < PIXELS_INTERPOLATION_SIZE * BATCH_COMPUTE_FEATURES_SIZE * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y) /4; ++i) {

			// i goes over [BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][BATCH_PIXELS_SIZE_/4] array so get indexes for both manually
			int f = indexing.getIndex<0>(i);
			int interpolation_j = indexing.getIndex<1>(i);
			int interpolation_i = indexing.getIndex<2>(i);
			int px = indexing.getIndex<3>(i);
			// since we store weight and offset into float4/int4 we need a proper index to access array of quad vectors
			int f_quad_index = f/4;

			int px_x = px % (BATCH_PIXELS_BY_WIDTH ? BATCH_PIXELS_SIZE_X/4 : BATCH_PIXELS_SIZE_X);
			int px_y = px / (BATCH_PIXELS_BY_WIDTH ? BATCH_PIXELS_SIZE_X/4 : BATCH_PIXELS_SIZE_X);

			// since array batches 4 pixels in float4 then get actual px address by multiplying with 4
			px_x = px_x * (BATCH_PIXELS_BY_WIDTH ?  4 : 1);
			px_y = px_y * (BATCH_PIXELS_BY_WIDTH ?  1 : 4);

			//if (interpolation_j > 0 || interpolation_i > 0)
			//	continue;

			// add interpolation offset to px_x and px_y
			px_x = px_x; + interpolation_i;
			px_y = px_y; + interpolation_j;

			// load data for next loop
			if (load_data.enabled) {

				int data_address_index = f_quad_index;
				int data_quad_index = f % 4;

				if (BATCH_PIXELS_BY_WIDTH) {
					//printf("loading data from address: %llu for f:%d, px_x: %d, px_y: %d and px: %d\n", load_data.address[f_quad_index].quad[f % 4] + px_x + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH, f, px_x, px_y, px);
					load_data.output[i] = reinterpret_cast<float4*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH)[0];
					/*
					load_data.output[i].x = 1;
					load_data.output[i].y = 1;
					load_data.output[i].z = 1;
					load_data.output[i].w = 1;
					load_data.output[i].x = *( float*)(load_data.address[data_address_index].quad[data_quad_index] + 0 + px_x + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH);
					load_data.output[i].y = *( float*)(load_data.address[data_address_index].quad[data_quad_index] + 1 + px_x + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH);
					load_data.output[i].z = *( float*)(load_data.address[data_address_index].quad[data_quad_index] + 2 + px_x + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH);
					load_data.output[i].w = *( float*)(load_data.address[data_address_index].quad[data_quad_index] + 3 + px_x + (px_y) * BlockSharedMemoryT::PITCHED_WIDTH);
					*/
				} else {
/*
					load_data.output[i].x = *( float*)(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 0) * BlockSharedMemoryT::PITCHED_WIDTH);
					load_data.output[i].y = *( float*)(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 1) * BlockSharedMemoryT::PITCHED_WIDTH);
					load_data.output[i].z = *( float*)(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 2) * BlockSharedMemoryT::PITCHED_WIDTH);
					load_data.output[i].w = *( float*)(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 3) * BlockSharedMemoryT::PITCHED_WIDTH);
*/
					load_data.output[i].x = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 0) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;
					load_data.output[i].y = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 1) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;
					load_data.output[i].z = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 2) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;
					load_data.output[i].w = reinterpret_cast<float2*>(load_data.address[data_address_index].quad[data_quad_index] + px_x + (px_y + 3) * BlockSharedMemoryT::PITCHED_WIDTH)[0].x;

				}
			}

			// compute for current loop
			if (compute.enabled) {
				// weights index must include interpolation index to get interpolation weight
				int weights_index = (interpolation_j * PIXELS_INTERPOLATION_Dx + interpolation_i) * BATCH_COMPUTE_FEATURES_SIZE/4 +  f_quad_index;

				// compute index must NOT include interpolation index since we sum all interpolation values into the same output
				int compute_index = f * ((BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4) + px;

				compute.output[compute_index].x += compute.weights[weights_index].x * compute.data[i].x;
				compute.output[compute_index].y += compute.weights[weights_index].y * compute.data[i].y;
				compute.output[compute_index].z += compute.weights[weights_index].z * compute.data[i].z;
				compute.output[compute_index].w += compute.weights[weights_index].w * compute.data[i].w;

			}
		}

		//__threadfence_block();
	}

};



template <typename BlockIndexingT>
__global__ void
fast_gauss_forward_pipeline_kernel(cudaTextureObject_t* filtered_images_tex, const float* filtered_images, const int* filter_offsets, const float* filter_weights, float* output,
							const int I, const int S, const int F, const int G,
							const int img_width_, const int img_height_,
							const int kernel_width, const int kernel_height) {

// INPUT: filtered images  	[I x S x H x W]
//		  filter offsets   	[F x S x G]
//		  filter weights   	[F x S x G]
// OUTPUT output  		 	[I x F x H x W]

#ifndef CUBIN_EMBEDDING

	typedef class BlockIndexingT::Kernel BlockIndexingKernel;

	static const int NUM_SM = BlockIndexingT::NUM_SM;
	static const int Bx = BlockIndexingT::Bx;
	static const int By = BlockIndexingT::By;
	static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
	static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
	static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
	static const int PIXELS_INTERPOLATION_Dx = BlockIndexingT::PIXELS_INTERPOLATION_Dx;
	static const int PIXELS_INTERPOLATION_Dy = BlockIndexingT::PIXELS_INTERPOLATION_Dy;
	static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
	static const int BATCH_COMPUTE_SUBFEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE;
	static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
	static const int BATCH_GAUSS_SIZE = BlockIndexingT::BATCH_GAUSS_SIZE;
	static const int IMG_WIDTH = 0; //BlockIndexingT::IMG_WIDTH;
	static const int IMG_HEIGHT = 0; //BlockIndexingT::IMG_HEIGHT; // may not be needed
	static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;

	static const int NUM_THREADS = BlockIndexingT::NUM_THREADS;

	static const bool BATCH_PIXELS_BY_WIDTH = BATCH_PIXELS_SIZE_X >= 4;

	// since we can load 4 weights and offsets from single LDS.128 we can batch 4 computes of features
	static const int BATCH_COMPUTE_FEATURES_SIZE = 4;

	// using float4 to load so use
	static const int BATCH_SH_PIXELS_SIZE = 4;

	static const int DOUBLE_BUFFERING = 2;

	static const int NUM_REPLICATE_OFFSETED = 1;

	static const int PIXELS_INTERPOLATION_SIZE = PIXELS_INTERPOLATION_Dx * PIXELS_INTERPOLATION_Dy;

	float* output_batch = reinterpret_cast<float*>(output);

	int img_width = img_width_; //IMG_WIDTH;
	int img_height = img_height_; //IMG_HEIGHT;

	BlockIndexingKernel block_indexing(img_width, img_height);

	int n = block_indexing.getImageIdx();

	int f_offset = block_indexing.getFeatureIdx();

	int f_block_idx = block_indexing.getFeatureBlockIdx();

	int block_width = block_indexing.getPosBlockSize().x;
	int block_height = block_indexing.getPosBlockSize().y;

	int block_x = block_indexing.getPosBlockIdx().x;
	int block_y = block_indexing.getPosBlockIdx().y;

	int thread_x = block_indexing.getPosThreadIdx().x;
	int thread_y = block_indexing.getPosThreadIdx().y;

	int G_MEM_SIZE = G / BATCH_GAUSS_SIZE;
	int S_MEM_SIZE = S / (BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE);
	int F_MEM_SIZE = F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES);

	static const int OFFSET_BLOCK_MEM_SIZE = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE *  BLOCK_FEATURES;
	static const int WEIGHT_BLOCK_MEM_SIZE = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_SIZE * BATCH_FEATURES_SIZE * BLOCK_FEATURES;


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


	typedef BlockSharedMemory<NUM_THREADS, BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_SIZE * BATCH_FEATURES_SIZE * BLOCK_FEATURES,
								1, 0, DOUBLE_BUFFERING, float4, BATCH_SH_PIXELS_SIZE> SharedMemWeights;
	typedef BlockSharedMemory<NUM_THREADS, BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE * BLOCK_FEATURES,
								1, 0, DOUBLE_BUFFERING, int4, BATCH_SH_PIXELS_SIZE> SharedMemOffsets;

#define ENABLE_SHARED_WEIGHTS_OFFSETS 1

#ifdef ENABLE_SHARED_WEIGHTS_OFFSETS
	__shared__ typename SharedMemWeights::Data data_weights;
	__shared__ typename SharedMemOffsets::Data data_offsets;

	SharedMemWeights weights_sh_class(data_weights, make_int2(thread_x, thread_y));
	SharedMemOffsets offsets_sh_class(data_offsets, make_int2(thread_x, thread_y));

	float4* weights_batch_sh = (float4*)weights_sh_class.getData(0);
	int4* offset_batch_sh = (int4*)offsets_sh_class.getData(0);
#endif



/*
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

		int index_w = 0;
		int index_off = 0;
		#pragma unroll
		for (int s = 0; s < BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE; ++s) {
			#pragma unroll
			for (int g = 0; g < BATCH_GAUSS_SIZE; ++g) {
				#pragma unroll
				for (int i = 0; i < PIXELS_INTERPOLATION_SIZE; ++i) {
					#pragma unroll
					for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {

						#pragma unroll
						for (int f_block = 0; f_block < BLOCK_FEATURES; ++f_block) {

							{
								if (i == 0) {
									offset_batch_sh[index_off].x = 0;
									offset_batch_sh[index_off].y = 0;
									offset_batch_sh[index_off].z = 0;
									offset_batch_sh[index_off].w = 0;

									index_off++;
								}
								weights_batch_sh[index_w].x = 0.25 / BATCH_GAUSS_SIZE;
								weights_batch_sh[index_w].y = 0.25 / BATCH_GAUSS_SIZE;
								weights_batch_sh[index_w].z = 0.25 / BATCH_GAUSS_SIZE;
								weights_batch_sh[index_w].w = 0.25 / BATCH_GAUSS_SIZE;


								index_w++;
							}
						}
					}
				}
			}
		}
	}
*/

	// WARNING: leave this part in otherwise it works slower !!! (probably due to some compiler optimization)
//	if (threadIdx.x == 100000000)
//		for (int i = 0; i < 2 * BATCH_MEM_SUBFEATURES_SIZE; ++i)
//			image_sh_class.template load_global<IMG_WIDTH + 2*MAX_OFFSET,0,true,0>(reinterpret_cast<float4*>(NULL), reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(i)) ) ;

	//__syncthreads();

	float4 out_val[BATCH_FEATURES_SIZE][(BATCH_PIXELS_SIZE_X*BATCH_PIXELS_SIZE_Y)/4];

	for (int f = 0; f < BATCH_FEATURES_SIZE ; ++f) {
		for (int px = 0; px < (BATCH_PIXELS_SIZE_X*BATCH_PIXELS_SIZE_Y)/4; ++px) {

			out_val[f][px].x = 0;
			out_val[f][px].y = 0;
			out_val[f][px].z = 0;
			out_val[f][px].w = 0;
		}
	}

	PipelineEngine<BATCH_PIXELS_SIZE_X,
					BATCH_PIXELS_SIZE_Y,
					BATCH_PIXELS_BY_WIDTH,
					PIXELS_INTERPOLATION_Dx,
					PIXELS_INTERPOLATION_Dy,
					BATCH_FEATURES_SIZE,
					BATCH_COMPUTE_FEATURES_SIZE,
					BATCH_COMPUTE_SUBFEATURES_SIZE,
					BATCH_MEM_SUBFEATURES_SIZE,
					BLOCK_FEATURES,
					IMG_WIDTH, IMG_HEIGHT,
					SharedMem> pipeline(image_sh_class);


	const int f_start_block = f_offset - f_block_idx;

	const float* _image_global_current = filtered_images + OFFSET(n,
																  0,
																  MAX_OFFSET + block_y + image_sh_class.getThreadIdx().y,
																  MAX_OFFSET + block_x + image_sh_class.getThreadIdx().x,
																  I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);

	const int* _filter_offset_current = filter_offsets +  OFFSET(f_start_block / (BLOCK_FEATURES*BATCH_FEATURES_SIZE),
																 0,
																 0,
																 0,
																 F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE);

	const float* _filter_weights_current = filter_weights + OFFSET(f_start_block/ (BLOCK_FEATURES*BATCH_FEATURES_SIZE) ,
																   0,
																   0,
																   0,
																   F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, WEIGHT_BLOCK_MEM_SIZE);

#ifdef ENABLE_SHARED_WEIGHTS_OFFSETS
	const int* _filter_offset_next = _filter_offset_current + offsets_sh_class.getThreadIdx().x;
	const float* _filter_weights_next = _filter_weights_current + weights_sh_class.getThreadIdx().x;

	if (1){


		// load offsets and weights for the first one
		offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,false,0>(reinterpret_cast<const int4*>(_filter_offset_current + offsets_sh_class.getThreadIdx().x),
																			   reinterpret_cast<int4*>(offsets_sh_class.getDataThreadIndexingWrite(0)));

		weights_sh_class.template load_global<WEIGHT_BLOCK_MEM_SIZE,0,false,1>(reinterpret_cast<const float4*>(_filter_weights_current + weights_sh_class.getThreadIdx().x),
																			   reinterpret_cast<float4*>(weights_sh_class.getDataThreadIndexingWrite(0)));

	}
#else
	const int* _filter_offset_next = _filter_offset_current;
	const float* _filter_weights_next = _filter_weights_current;
#endif
	if (1){
		// load first batch of subfeatures/input data into shared memory

		for (int s = 0 ; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {
			image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(_image_global_current + (s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
																					  	  	  	  	  	   reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(s % BATCH_MEM_SUBFEATURES_SIZE)),
																										   img_width + 2 * MAX_OFFSET);
		}
	}

	__syncthreads();

	const int MAX_S_OUTER_INDEX = S /  BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE;

	for (int s_offset_outer = 0; s_offset_outer < S; s_offset_outer+=BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE) {

		const int s_outer_index = s_offset_outer / (BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE);

		const int s_buffer_index = s_outer_index % DOUBLE_BUFFERING;

		const int* filter_offset_current  = _filter_offset_current + OFFSET(0, s_outer_index, 0, 0, F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE);
		const float* filter_weights_current  = _filter_weights_current + OFFSET(0, s_outer_index, 0, 0, F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, WEIGHT_BLOCK_MEM_SIZE);

		const int* filter_offset_next  = _filter_offset_next + OFFSET(0, s_outer_index + 1 , 0, 0, F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, OFFSET_BLOCK_MEM_SIZE);
		const float* filter_weights_next = _filter_weights_next + OFFSET(0, s_outer_index + 1, 0, 0, F_MEM_SIZE, S_MEM_SIZE, G_MEM_SIZE, WEIGHT_BLOCK_MEM_SIZE);

		const float* image_global_current = _image_global_current + OFFSET(0, s_outer_index, 0, 0, I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);
		const float* image_global_next = _image_global_current + OFFSET(0, s_outer_index + 1, 0, 0, I, S, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET);

		ptr4 off_A[BATCH_GAUSS_SIZE][BATCH_COMPUTE_FEATURES_SIZE/4],
			 off_B[BATCH_GAUSS_SIZE][BATCH_COMPUTE_FEATURES_SIZE/4];

		float4 w_A[BATCH_GAUSS_SIZE][PIXELS_INTERPOLATION_SIZE][BATCH_COMPUTE_FEATURES_SIZE/4],
			   w_B[BATCH_GAUSS_SIZE][PIXELS_INTERPOLATION_SIZE][BATCH_COMPUTE_FEATURES_SIZE/4];

		float4 d_A[BATCH_GAUSS_SIZE][BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4],
			   d_B[BATCH_GAUSS_SIZE][BATCH_FEATURES_SIZE][PIXELS_INTERPOLATION_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4];



		struct IterIndex {
			int s; // sub-feature index
			int f; // feature index
			int g; // gauss component index
		};


		// global loading is done imediately (no delay)
		// to simplyfiy the code for global loading we can force global loading to be done BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE loops before
		// other units start
		static const int start_delay_global_load = 1; //1;
		static const int start_delay_offset_load = 0;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
		static const int start_delay_w_load = 1;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
		static const int start_delay_data_load = 1;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;
		static const int start_delay_compute = 2;// + BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;

		// NOTE: EXTRA_LOOPS is max value out of start_delay_global_load, start_delay_offset_load, start_delay_w_load, start_delay_data_load and start_delay_compute
		static const int EXTRA_LOOPS = MAX(start_delay_global_load,
											MAX(start_delay_offset_load,
												MAX(start_delay_w_load,
													MAX(start_delay_data_load, start_delay_compute))));

		int NUM_ITER = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;


		// iterations go over subsets of [S x F ] i.e. [BATCH_MEM_SUBFEATURES_SIZE] * [BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE]

		NDIndexing<BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE,
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
			pipeline.load_weights.enabled = pipeline.should_run(index, start_delay_w_load, NUM_ITER);
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

				// we actually load next batch of subfeatures so add BATCH_MEM_SUBFEATURES_SIZE
				load_global.s = load_global.s + BATCH_MEM_SUBFEATURES_SIZE;

				if (load_global_enabled)
					load_global_enabled = load_global.f == 0 && load_global.g == 0;

				// TODO: do not load if this is last s_offset_outer index

				int double_buffer_index = (s_buffer_index + load_global.s/BATCH_MEM_SUBFEATURES_SIZE) % 2;
				int subfeat_buffer_index = load_global.s % BATCH_MEM_SUBFEATURES_SIZE;

				// if this is last iteration before moving to next s_offset_outer index then load for the next one
				bool load_next_s_outer = load_global.s < BATCH_MEM_SUBFEATURES_SIZE * BATCH_COMPUTE_SUBFEATURES_SIZE;

				const float* image_global_load =  load_next_s_outer ? image_global_current : image_global_next;

				load_global.s = load_global.s % (BATCH_MEM_SUBFEATURES_SIZE * BATCH_COMPUTE_SUBFEATURES_SIZE);

				// NOTE: we do not need to check for the last index since data should be padded with one extra image to prevent reading invalid data
				//if (load_next_s_outer && s_outer_index >= MAX_S_OUTER_INDEX)
				//	load_global_enabled = false;

				global_d = double_buffer_index;

				int buffer_index = OFFSET(0, double_buffer_index, subfeat_buffer_index, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

				// load global
				if (load_global_enabled) {
					image_sh_class.template load_global<IMG_WIDTH + 2 * MAX_OFFSET,NUM_REPLICATE_OFFSETED,false,1>(reinterpret_cast<const float4*>(image_global_load + (load_global.s) * (img_width + 2*MAX_OFFSET) * (img_height + 2*MAX_OFFSET)),
																												   reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(buffer_index)),
																												   img_width + 2 * MAX_OFFSET);
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
					}
				}

				shared_d_next = next_double_buffer_index;

				// switch between registers every iteration
				bool use_reg_A = (index - start_delay_offset_load) % 2 == 0 ? true : false;

				load_offset_reg_A = use_reg_A;

				shared_d_off = double_buffer_index;

				int address_off = OFFSET(load_offset_index.s, load_offset_index.g, load_offset_index.f/4, f_block_idx/BATCH_FEATURES_SIZE, BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE, BATCH_GAUSS_SIZE, BATCH_FEATURES_SIZE/4, BLOCK_FEATURES);

				// load offset
#ifdef ENABLE_SHARED_WEIGHTS_OFFSETS
				pipeline.load_offset.offset_address = offset_batch_sh + address_off + (s_offset_outer % DOUBLE_BUFFERING) * OFFSET_BLOCK_MEM_SIZE/4;
				//pipeline.load_offset.offset_address = &reinterpret_cast<int4*>((int*)filter_offset_current)[address_off];

				//pipeline.load_offset.offset_address = use_reg_A ?
				//										offset_batch_sh + address_off + (s_offset_outer % DOUBLE_BUFFERING) * OFFSET_BLOCK_MEM_SIZE/4
				//										: &reinterpret_cast<int4*>((int*)filter_offset_current)[address_off];
				//pipeline.load_offset.offset_address = &offset_batch_sh[load_offset_index.s][load_offset_index.g][0][load_offset_index.f/4][f_block_idx/BATCH_FEATURES_SIZE];
#else
				pipeline.load_offset.offset_address = &reinterpret_cast<int4*>((int*)filter_offset_current)[address_off];
#endif

				int buffer_index = OFFSET(0, double_buffer_index, subfeat_buffer_index, 0, 1, DOUBLE_BUFFERING, BATCH_MEM_SUBFEATURES_SIZE, NUM_REPLICATE_OFFSETED+1);

				pipeline.load_offset.base_address = image_sh_class.getDataThreadIndexingRead(buffer_index);
				pipeline.load_offset.output = (ptr4*)(use_reg_A ? &off_A[load_offset_index.g][0] : &off_B[load_offset_index.g][0]);

			}
			bool load_w_reg_A;
			{
				// w and data loading is done with single delay

				load_w_index.s = indexing.getIndex<0>(index - start_delay_w_load);
				load_w_index.g = indexing.getIndex<1>(index - start_delay_w_load);
				load_w_index.f = indexing.getIndex<2>(index - start_delay_w_load) * BATCH_COMPUTE_FEATURES_SIZE;
				// switch between registers every iteration
				bool use_reg_A = (index - start_delay_w_load) % 2 == 0 ? true : false;

				load_w_reg_A = use_reg_A;

				int address_off = OFFSET5(load_w_index.s, load_w_index.g, 0, load_w_index.f/4, f_block_idx/BATCH_FEATURES_SIZE, BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE, BATCH_GAUSS_SIZE, PIXELS_INTERPOLATION_SIZE, BATCH_FEATURES_SIZE/4, BLOCK_FEATURES);

				// load w
#ifdef ENABLE_SHARED_WEIGHTS_OFFSETS
				//pipeline.load_weights.address = weights_batch_sh + address_off + (s_offset_outer % 2) * WEIGHT_BLOCK_MEM_SIZE/4;
				//pipeline.load_weights.address = &reinterpret_cast<float4*>((float*)filter_weights_current)[address_off];

				// we can utilize texture/L1 cache and load every second one from global data
				pipeline.load_weights.address = (index - start_delay_w_load) % 4 == 1 ?
													weights_batch_sh +  address_off + (s_offset_outer % 2) * WEIGHT_BLOCK_MEM_SIZE/4
													: &reinterpret_cast<float4*>((float*)filter_weights_current)[address_off];

				//pipeline.load_weights.address = &weights_batch_sh[load_w_index.s][load_w_index.g][0][load_w_index.f/4][f_block_idx/BATCH_FEATURES_SIZE];
#else
				pipeline.load_weights.addre2ss = &reinterpret_cast<float4*>((float*)filter_weights_current)[address_off];
#endif
				pipeline.load_weights.output = (float4*)(use_reg_A ? w_A[load_w_index.g][0] : w_B[load_w_index.g][0]);

			}
			bool load_data_reg_A;
			{

				load_data_index.s = indexing.getIndex<0>(index - start_delay_data_load);
				load_data_index.g = indexing.getIndex<1>(index - start_delay_data_load);
				load_data_index.f = indexing.getIndex<2>(index - start_delay_data_load) * BATCH_COMPUTE_FEATURES_SIZE;

				// switch between registers every iteration
				bool use_reg_A = (index - start_delay_data_load) % 2 == 0 ? true : false;

				//shared_d_current = current_double_buffer_index;

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
				pipeline.compute.weights = (float4*)(use_reg_A ? w_A[compute_index.g][0] : w_B[compute_index.g][0]);
				pipeline.compute.data = (float4*)(use_reg_A ? d_A[compute_index.g][compute_index.f] : d_B[compute_index.g][compute_index.f]);
				pipeline.compute.output = out_val[compute_index.f];

			}


			// sync only before data buffer is switched
			if (require_sync) {
				// NOTE: sync is not needed if we have more then enough operations to cover the latency of sore operations
				// we can rughly say that if there is more then 128 operations then STS latency should be hidden (STS latency should not be more then 100 operations on different platforms)
				// however since store may be issued half way through operations then use 512 operations as limit
				if (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y * BATCH_FEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_COMPUTE_SUBFEATURES_SIZE < 512)
					__syncthreads();
			}
/*
			if (threadIdx.x == 0 && blockIdx.x == 0) {
				if (require_sync)
					printf("iter: %d, sycned\n", index);

				printf("pipeline iter %d (g:%d): gl %d (s:%d, f:%d, buff:%d, next buff:%d; reg:%d), off %d (s:%d, f:%d, reg:%d, buff:%d), w %d (s:%d, f:%d, reg:%d), data %d (s:%d, f:%d, buff:%d, next buff:%d; reg:%d), compute %d (s:%d, f:%d, reg:%d)\n",
						index, g,
						(int)pipeline.load_global.enabled, load_global.s, load_global.f, global_d,
						(int)pipeline.load_offset.enabled, load_offset_index.s, load_offset_index.f, shared_d_off, shared_d_next, (int)load_offset_reg_A,
						(int)pipeline.load_weights.enabled, load_w_index.s, load_w_index.f, (int)load_w_reg_A,
						(int)pipeline.load_data.enabled, load_data_index.s, load_data_index.f, shared_d_current, -1, (int)load_data_reg_A,
						(int)pipeline.compute.enabled, compute_index.s, compute_index.f, (int)compute_reg_A);


			}
*/
			pipeline.execute_step();


#ifdef ENABLE_SHARED_WEIGHTS_OFFSETS
			// if next iteration is not the last one then load offsets and weights for the next one - using double buffering so we do not intereput computation of the current one
			if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index + 4 ==  NUM_ITER + EXTRA_LOOPS )
			//if (index + 4 ==  NUM_ITER + EXTRA_LOOPS )

			//if (s_outer_index + 1 < MAX_S_OUTER_INDEX && index == 0 && 0)
			// NOTE: we do not need to check for the last index since data should be padded with one extra image to prevent reading invalid data
			if (index == 0)
			{

				{ // TODO: split load_global into two function one for load and one for read to force distance between LDG and STS commands

					offsets_sh_class.template load_global<OFFSET_BLOCK_MEM_SIZE,0,false,1>(reinterpret_cast<const int4*>(filter_offset_next),
																					 	   reinterpret_cast<int4*>(offsets_sh_class.getDataThreadIndexingWrite((s_offset_outer + 1 ) % 2)));

					weights_sh_class.template load_global<WEIGHT_BLOCK_MEM_SIZE,0,false,1>(reinterpret_cast<const float4*>(filter_weights_next),
																					 	   reinterpret_cast<float4*>(weights_sh_class.getDataThreadIndexingWrite((s_offset_outer + 1 ) % 2)));

				}

			}
#endif

/*
			if (threadIdx.x == 0 && blockIdx.x == 0) {
			//if (compute_index.f == (BATCH_FEATURES_SIZE/4 -1)*4)

				if (pipeline.compute.enabled) {
					for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
						for (int px_y = 0; px_y < BATCH_PIXELS_SIZE_Y; ++px_y) {
							for (int px_x = 0; px_x < BATCH_PIXELS_SIZE_X; px_x+=4) {
								float4 tmp = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/4];
								printf("output vals: %f, %f, %f, %f\n", tmp.x, tmp.y, tmp.z, tmp.w);

							}
						}
					}
				}
			}
*/
		}
	}


	// TODO: we can perform shuffle between output registers and ensure only coalesed output using only STG.128
	#pragma unroll
	for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
		if (BATCH_PIXELS_BY_WIDTH) {
			// version for loading per 4 pixels by width and 1 pixel per height
			#pragma unroll
			for (int px_y = 0; px_y < BATCH_PIXELS_SIZE_Y; ++px_y) {
				#pragma unroll
				for (int px_x = 0; px_x < BATCH_PIXELS_SIZE_X; px_x+=4) {
					//float4 tmp;
					//tmp.x = 8; tmp.y = 8; tmp.z = 8; tmp.w = 8;
					//reinterpret_cast<float4*>(output_batch)[OFFSET(n, f + f_offset, (block_y + thread_y + px_y), (block_x + thread_x + px_x), I, F, img_height, img_width + 2*MAX_OFFSET)/4] = tmp;
					reinterpret_cast<float4*>(output_batch)[OFFSET(n, f + f_offset, (block_y + thread_y + px_y), (block_x + thread_x + px_x), I, F, img_height + 2*MAX_OFFSET, img_width + 2*MAX_OFFSET)/4] = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/4];
				}
			}
		} else {
			// version for loading per 1 pixels by width and 4 pixel per height
			#pragma unroll
			for (int px_y = 0; px_y < BATCH_PIXELS_SIZE_Y; px_y+=4) {
				#pragma unroll
				for (int px_x = 0; px_x < BATCH_PIXELS_SIZE_X; ++px_x) {
					//if (threadIdx.x == 0 && blockIdx.x == 0)
					//	printf("output at f = %d, px_x = %d, px_y = %d, px index :%d\n", f, px_x, px_y, px_y/4 * BATCH_PIXELS_SIZE_X + px_x);

					output_batch[OFFSET(n, f + f_offset, (block_y + thread_y + 0 + px_y), (block_x + thread_x + px_x), I, F, img_height, img_width)] = out_val[f][px_y/4 * BATCH_PIXELS_SIZE_X + px_x].x;
					output_batch[OFFSET(n, f + f_offset, (block_y + thread_y + 1 + px_y), (block_x + thread_x + px_x), I, F, img_height, img_width)] = out_val[f][px_y/4 * BATCH_PIXELS_SIZE_X + px_x].y;
					output_batch[OFFSET(n, f + f_offset, (block_y + thread_y + 2 + px_y), (block_x + thread_x + px_x), I, F, img_height, img_width)] = out_val[f][px_y/4 * BATCH_PIXELS_SIZE_X + px_x].z;
					output_batch[OFFSET(n, f + f_offset, (block_y + thread_y + 3 + px_y), (block_x + thread_x + px_x), I, F, img_height, img_width)] = out_val[f][px_y/4 * BATCH_PIXELS_SIZE_X + px_x].w;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// HACK: inject this code to force compiler to use around 110-128 registers (N=95 gets around 110 registers)
	// (by default compiler agressively optimiztes num of registers even if it could affort more registers
/*
	if (threadIdx.y == 11111111) {
		float4* image_sh = (float4*)image_sh_class.getData(0);

		static const int N = 100;
		volatile float4 dd[N/4];
		for (int i = 0; i < N/4; ++i) {
			dd[i].x = i*1;
			dd[i].y = i*2;
			dd[i].z = i*3;
			dd[i].w = i*4;
		}

		__threadfence_block();

		for (int i = 0; i < N/4; ++i) {
			image_sh[i].x = dd[i].x * dd[N/4-1 - i].x;
			image_sh[i].y = dd[i].y * dd[N/4-1 - i].y;
			image_sh[i].z = dd[i].z * dd[N/4-1 - i].z;
			image_sh[i].w = dd[i].w * dd[N/4-1 - i].w;
		}
	}
*/
#endif
}



template <typename BlockIndexingT>
__global__ void
fast_gauss_forward_kernel(cudaTextureObject_t* filtered_images_tex, const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_weights, float* output,
							const int I, const int S, const int F, const int G,
							const int img_width_, const int img_height_,
							const int kernel_width, const int kernel_height) {

// INPUT: filtered images  	[I x S x H x W]
//		  filter offsets   	[F x S x G]
//		  filter weights   	[F x S x G]
// OUTPUT output  		 	[I x F x H x W]


#ifndef CUBIN_EMBEDDING

	typedef class BlockIndexingT::Kernel BlockIndexingKernel;

	static const int NUM_SM = BlockIndexingT::NUM_SM;
	static const int Bx = BlockIndexingT::Bx;
	static const int By = BlockIndexingT::By;
	static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
	static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
	static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
	static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
	static const int BATCH_COMPUTE_SUBFEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE;
	static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
	static const int BATCH_GAUSS_SIZE = BlockIndexingT::BATCH_GAUSS_SIZE;
	static const int IMG_WIDTH = BlockIndexingT::IMG_WIDTH;
	static const int IMG_HEIGHT = BlockIndexingT::IMG_HEIGHT;
	static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;

	static const int NUM_THREADS = BlockIndexingT::NUM_THREADS;

	static const int BATCH_PIXELS_SIZE = BATCH_PIXELS_SIZE_Y;


	static const int BATCH_SH_PIXELS_SIZE = 4;

	int img_width = IMG_WIDTH;
	int img_height = IMG_HEIGHT;

	BlockIndexingKernel block_indexing(img_width, img_height);

	int n = block_indexing.getImageIdx();

	int f_offset = block_indexing.getFeatureIdx();

	int f_block_idx = block_indexing.getFeatureBlockIdx();

	int block_width = block_indexing.getPosBlockSize().x;
	int block_height = block_indexing.getPosBlockSize().y;

	int block_x = block_indexing.getPosBlockIdx().x;
	int block_y = block_indexing.getPosBlockIdx().y;

	int thread_x = block_indexing.getPosThreadIdx().x;
	int thread_y = block_indexing.getPosThreadIdx().y;

	typedef BlockSharedMemory<NUM_THREADS, Bx, By * BATCH_PIXELS_SIZE, MAX_OFFSET, 2 * BATCH_MEM_SUBFEATURES_SIZE, float4, BATCH_SH_PIXELS_SIZE> SharedMem;

	__shared__ typename SharedMem::Data data;

	SharedMem image_sh_class(data, make_int2(thread_x,thread_y));


	float* output_batch = reinterpret_cast<float*>(output);

	float4* output_batch4 = reinterpret_cast<float4*>(output);

	float* image_sh = image_sh_class.getData(0 * BATCH_MEM_SUBFEATURES_SIZE);
	float* image_sh_buf_B = image_sh_class.getData(1 * BATCH_MEM_SUBFEATURES_SIZE);

	__shared__ int4 offset_x_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];
	__shared__ int4 offset_y_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];
	__shared__ float4 weights_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

		#pragma unroll
		for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

			#pragma unroll
			for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {
				{
					offset_x_batch_sh[s][f].x = 0;
					offset_x_batch_sh[s][f].y = 0;
					offset_x_batch_sh[s][f].z = 0;
					offset_x_batch_sh[s][f].w = 0;

					offset_y_batch_sh[s][f].x = 0;
					offset_y_batch_sh[s][f].y = 0;
					offset_y_batch_sh[s][f].z = 0;
					offset_y_batch_sh[s][f].w = 0;

					weights_batch_sh[s][f].x = 1;
					weights_batch_sh[s][f].y = 1;
					weights_batch_sh[s][f].z = 1;
					weights_batch_sh[s][f].w = 1;
				}
			}
		}
	}

	__syncthreads();


	int thread_sh_x = image_sh_class.getThreadIdx().x;
	int thread_sh_y = image_sh_class.getThreadIdx().y;

	if (1){

		for (int i = 0; i < 2 * BATCH_MEM_SUBFEATURES_SIZE; ++i)
			image_sh_class.template load_global<IMG_WIDTH,0,true,0>(reinterpret_cast<float4*>(NULL), reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(i)) ) ;

	}


	__syncthreads();

	cudaTextureObject_t current_image_tex = filtered_images_tex[n];

	float4 out_val[BATCH_FEATURES_SIZE][BATCH_PIXELS_SIZE/4];

	for (int f = 0; f < BATCH_FEATURES_SIZE ; ++f) {
		for (int px = 0; px < BATCH_PIXELS_SIZE/4; ++px) {

			out_val[f][px].x = 0;
			out_val[f][px].y = 0;
			out_val[f][px].z = 0;
			out_val[f][px].w = 0;
		}
	}

	// for reading and compute
	float* image_sh_at_thread = image_sh + image_sh_class.template getOffsetAt<float>(MAX_OFFSET + thread_x, MAX_OFFSET + thread_y);

	// for copying from global to shared
	float* image_sh_at_thread_sh = image_sh + image_sh_class.template getOffsetAt<float>(thread_sh_x, thread_sh_y);


	const float* filtered_image_n_at_thread_read = filtered_images + OFFSET(n, 0, block_y + image_sh_class.getThreadIdx().y, block_x + image_sh_class.getThreadIdx().x, I, S, img_height, img_width);

	for (int s_offset_outer = 0; s_offset_outer < S; s_offset_outer+=BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE) {

		const float* filtered_image_s_offset_at_thread_read = filtered_image_n_at_thread_read +  s_offset_outer * (img_height * img_width);

		#pragma unroll
		for (int s_index = 0 ; s_index < BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE; s_index+=BATCH_MEM_SUBFEATURES_SIZE) {

			int4 offset_x_batch[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];
			float4 weights_batch[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];

			//if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
			{

				#pragma unroll
				for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

					#pragma unroll
					for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {
						//  TODO: use multiple threads to load values effectivly only one instruction to load offset and weigts !!
						//offset_x_batch_sh[s][f] = reinterpret_cast<const int4*>(filter_offsets_x)[(s_offset_outer + s_index + s) * (BATCH_FEATURES_SIZE/4) + f];
						//weights_batch_sh[s][f] = reinterpret_cast<const float4*>(filter_weights)[(s_offset_outer + s_index + s) * (BATCH_FEATURES_SIZE/4) + f];
						offset_x_batch[s][f] = reinterpret_cast<const int4*>(filter_offsets_x)[(s_offset_outer + s_index + s) * (BATCH_FEATURES_SIZE/4) + f];
						weights_batch[s][f] = reinterpret_cast<const float4*>(filter_weights)[(s_offset_outer + s_index + s) * (BATCH_FEATURES_SIZE/4) + f];
					}
				}
			}

			#pragma unroll
			for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

				float4* read_address =  reinterpret_cast<float4*>((float*)filtered_image_s_offset_at_thread_read + (s_index + s) * IMG_WIDTH * IMG_HEIGHT);

				float4* write_address = reinterpret_cast<float4*>(image_sh_at_thread_sh + s*(SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT));


				image_sh_class.template load_global<IMG_WIDTH,0,false,1>(read_address, write_address);

			}

			// make sure all threads have finished loading memory
			__syncthreads();

//			image_sh_class.print();


			#pragma unroll
			for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE ; ++s) {
				// position subfeature pointer to start in shared memory
				const void* s_image_sh = image_sh_at_thread + (s)*(SharedMem::PITCHED_WIDTH*SharedMem::ALLOC_HEIGHT);

				#pragma unroll
				for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {

					int4 offset_x;
					int4 offset_y;

					float4 w;
					offset_x.x = offset_x_batch[s][f].x;
					offset_x.y = offset_x_batch[s][f].y;
					offset_x.z = offset_x_batch[s][f].z;
					offset_x.w = offset_x_batch[s][f].w;

					w.x = weights_batch[s][f].x;
					w.y = weights_batch[s][f].y;
					w.z = weights_batch[s][f].z;
					w.w = weights_batch[s][f].w;

					/*
					offset_x.x = offset_x_batch_sh[s][f].x;
					offset_x.y = offset_x_batch_sh[s][f].y;
					offset_x.z = offset_x_batch_sh[s][f].z;
					offset_x.w = offset_x_batch_sh[s][f].w;

					w.x = weights_batch_sh[s][f].x;
					w.y = weights_batch_sh[s][f].y;
					w.z = weights_batch_sh[s][f].z;
					w.w = weights_batch_sh[s][f].w;
					 	*/

					//for (int g; g < G; ++g)
					int g = 0;
					{
						float4 value;
						int4 sh_image_offset;
						sh_image_offset.x = offset_x.x;
						sh_image_offset.y = offset_x.y;
						sh_image_offset.z = offset_x.z;
						sh_image_offset.w = offset_x.w;

						for (int px = 0; px < BATCH_PIXELS_SIZE_Y/4; ++px) {
							// use void* pointer to avoid doing another multiplation by sizeof(float) and instead embeed this value directly into offsets
							out_val[f*4 + 0][px].x +=  w.x *  ((float*)(s_image_sh + sh_image_offset.x + (0 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 0][px].y +=  w.x *  ((float*)(s_image_sh + sh_image_offset.x + (1 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 0][px].z +=  w.x *  ((float*)(s_image_sh + sh_image_offset.x + (2 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 0][px].w +=  w.x *  ((float*)(s_image_sh + sh_image_offset.x + (3 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];

							out_val[f*4 + 1][px].x +=  w.y *  ((float*)(s_image_sh + sh_image_offset.y + (0 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 1][px].y +=  w.y *  ((float*)(s_image_sh + sh_image_offset.y + (1 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 1][px].z +=  w.y *  ((float*)(s_image_sh + sh_image_offset.y + (2 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 1][px].w +=  w.y *  ((float*)(s_image_sh + sh_image_offset.y + (3 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];

							out_val[f*4 + 2][px].x +=  w.z *  ((float*)(s_image_sh + sh_image_offset.z + (0 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 2][px].y +=  w.z *  ((float*)(s_image_sh + sh_image_offset.z + (1 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 2][px].z +=  w.z *  ((float*)(s_image_sh + sh_image_offset.z + (2 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 2][px].w +=  w.z *  ((float*)(s_image_sh + sh_image_offset.z + (3 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];

							out_val[f*4 + 3][px].x +=  w.w *  ((float*)(s_image_sh + sh_image_offset.w + (0 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 3][px].y +=  w.w *  ((float*)(s_image_sh + sh_image_offset.w + (1 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 3][px].z +=  w.w *  ((float*)(s_image_sh + sh_image_offset.w + (2 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];
							out_val[f*4 + 3][px].w +=  w.w *  ((float*)(s_image_sh + sh_image_offset.w + (3 + px * BATCH_PIXELS_SIZE_Y) * SharedMem::PITCHED_WIDTH * sizeof(float)))[0];

						}
					}

				}
			}
		}
	}

	// TODO: we can perform shuffle between output registers and ensure only coalesed output using only STG.128
	#pragma unroll
	for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
		#pragma unroll
		for (int px = 0; px < BATCH_PIXELS_SIZE; px+=4) {
			output_batch[OFFSET(n, f + f_offset, (block_y + thread_y + 0 + px), (block_x + thread_x), I, F, img_height, img_width)] = out_val[f][px/4].x;
			output_batch[OFFSET(n, f + f_offset, (block_y + thread_y + 1 + px), (block_x + thread_x), I, F, img_height, img_width)] = out_val[f][px/4].y;
			output_batch[OFFSET(n, f + f_offset, (block_y + thread_y + 2 + px), (block_x + thread_x), I, F, img_height, img_width)] = out_val[f][px/4].z;
			output_batch[OFFSET(n, f + f_offset, (block_y + thread_y + 3 + px), (block_x + thread_x), I, F, img_height, img_width)] = out_val[f][px/4].w;
		}
	}

#endif
}



template <int Bx, int By, int BATCH_PIXELS_SIZE, int BATCH_FEATURES_SIZE, int BATCH_COMPUTE_SUBFEATURES_SIZE, int BATCH_MEM_SUBFEATURES_SIZE, int IMG_WIDTH, int IMG_HEIGHT, int MAX_OFFSET>
__global__ void
fast_gauss_forward_LDS128_kernel(cudaTextureObject_t* filtered_images_tex, const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_weights, float* output,
							const int I, const int S, const int F, const int G,
							const int img_width_, const int img_height_,
							const int kernel_width, const int kernel_height) {

// INPUT: filtered images  	[I x S x H x W]
//		  filter offsets   	[F x S x G]
//		  filter weights   	[F x S x G]
// OUTPUT output  		 	[I x F x H x W]


#ifndef CUBIN_EMBEDDING

	static const int BATCH_SH_PIXELS_SIZE = 4;

	static const int NUM_THREADS = Bx * By;

	int img_width = IMG_WIDTH;
	int img_height = IMG_HEIGHT;

	float* output_batch = reinterpret_cast<float*>(output);

	float4* output_batch4 = reinterpret_cast<float4*>(output);

	int f_offset =  blockIdx.x * BATCH_FEATURES_SIZE;


	int threadIdx_x = threadIdx.x % (Bx);
	int threadIdx_y = threadIdx.x / (Bx);

	int blockIdx_x = blockIdx.y % (img_width / WARP_SIZE);
	int blockIdx_y = blockIdx.y / (img_width / WARP_SIZE);

	int block_width = Bx * BATCH_PIXELS_SIZE;
	int block_height = By ;

	int block_x = BATCH_PIXELS_SIZE * (blockIdx_x * Bx);
	int block_y = (blockIdx_y * By);

	int thread_x = BATCH_PIXELS_SIZE * threadIdx_x;
	int thread_y = threadIdx_y;
	int n = blockIdx.z;


	typedef BlockSharedMemory<NUM_THREADS, Bx * BATCH_PIXELS_SIZE, By, MAX_OFFSET, 1 * BATCH_MEM_SUBFEATURES_SIZE, float4, BATCH_SH_PIXELS_SIZE> SharedMem;

	__shared__ typename SharedMem::Data data;

	SharedMem image_sh_class(data, make_int2(thread_x, thread_y));


	int thread_sh_x = image_sh_class.getThreadIdx().x;
	int thread_sh_y = image_sh_class.getThreadIdx().y;


	float* image_sh = image_sh_class.getData(0 * BATCH_MEM_SUBFEATURES_SIZE);

	__shared__ int4 offset_x_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];
	__shared__ int4 offset_y_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];
	__shared__ float4 weights_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

		#pragma unroll
		for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

			#pragma unroll
			for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {
				{
					offset_x_batch_sh[s][f].x = 0;
					offset_x_batch_sh[s][f].y = 0;
					offset_x_batch_sh[s][f].z = 0;
					offset_x_batch_sh[s][f].w = 0;

					offset_y_batch_sh[s][f].x = 0;
					offset_y_batch_sh[s][f].y = 0;
					offset_y_batch_sh[s][f].z = 0;
					offset_y_batch_sh[s][f].w = 0;

					weights_batch_sh[s][f].x = 1;
					weights_batch_sh[s][f].y = 1;
					weights_batch_sh[s][f].z = 1;
					weights_batch_sh[s][f].w = 1;
				}
			}
		}
	}

	__syncthreads();


	if (1){

		for (int i = 0; i < 1 * BATCH_MEM_SUBFEATURES_SIZE; ++i)
			image_sh_class.template load_global<IMG_WIDTH,0,true,0>(reinterpret_cast<float4*>(NULL), reinterpret_cast<float4*>(image_sh_class.getDataThreadIndexingWrite(i)) ) ;

	}

	__syncthreads();

	cudaTextureObject_t current_image_tex = filtered_images_tex[n];

	float4 out_val[BATCH_FEATURES_SIZE][BATCH_PIXELS_SIZE/4];

	for (int f = 0; f < BATCH_FEATURES_SIZE ; ++f) {
		for (int px = 0; px < BATCH_PIXELS_SIZE/4; ++px) {

			out_val[f][px].x = 0;
			out_val[f][px].y = 0;
			out_val[f][px].z = 0;
			out_val[f][px].w = 0;
		}
	}

	// for reading and compute
	float* image_sh_at_thread = image_sh + image_sh_class.template getOffsetAt<float>(MAX_OFFSET + thread_x, MAX_OFFSET + thread_y);

	// for copying from global to shared
	float* image_sh_at_thread_sh = image_sh + image_sh_class.template getOffsetAt<float>(thread_sh_x, thread_sh_y);


	//const float* filtered_image_n = filtered_images + OFFSET(n, 0, 0, 0, I, S, img_height, img_width);
	const float* filtered_image_n_at_thread_read = filtered_images + OFFSET(n, 0, block_y + image_sh_class.getThreadIdx().y, block_x + image_sh_class.getThreadIdx().x, I, S, img_height, img_width);

	for (int s_offset_outer = 0; s_offset_outer < S; s_offset_outer+=BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE) {

		//const float* filtered_image_s_offset = filtered_image_n +  s_offset_outer * ( img_height * img_width);
		const float* filtered_image_s_offset_at_thread_read = filtered_image_n_at_thread_read +  s_offset_outer * (img_height * img_width);

		#pragma unroll
		for (int s_index = 0 ; s_index < BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE; s_index+=BATCH_MEM_SUBFEATURES_SIZE) {

			int4 offset_x_batch[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];
			float4 weights_batch[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];

			//if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 1000)
			{

				#pragma unroll
				for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

					#pragma unroll
					for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {
					//int f = 0; {
						//  TODO: use multiple threads to load values effectivly only one instruction to load offset and weigts !!
						//offset_x_batch_sh[s][f] = reinterpret_cast<const int4*>(filter_offsets_x)[(s_offset_outer + s_index + s) * (BATCH_FEATURES_SIZE/4) + f];
						//weights_batch_sh[s][f] = reinterpret_cast<const float4*>(filter_weights)[(s_offset_outer + s_index + s) * (BATCH_FEATURES_SIZE/4) + f];
						offset_x_batch[s][f] = reinterpret_cast<const int4*>(filter_offsets_x)[(s_offset_outer + s_index + s) * (BATCH_FEATURES_SIZE/4) + f];
						weights_batch[s][f] = reinterpret_cast<const float4*>(filter_weights)[(s_offset_outer + s_index + s) * (BATCH_FEATURES_SIZE/4) + f];
					}
				}
			}

			#pragma unroll
			for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

				float4* read_address =  reinterpret_cast<float4*>((float*)filtered_image_s_offset_at_thread_read + (s_index + s) * IMG_WIDTH * IMG_HEIGHT);

				float4* write_address = reinterpret_cast<float4*>(image_sh_at_thread_sh + s*(SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT));

				image_sh_class.template load_global<IMG_WIDTH,0,false,1>(read_address, write_address);
			}

			// make sure all threads have finished loading memory
			__syncthreads();

//			image_sh_class.print();


			#pragma unroll
			for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE ; ++s) {
				// position subfeature pointer to start in shared memory
				const void* s_image_sh = image_sh_at_thread + (s)*(SharedMem::PITCHED_WIDTH*SharedMem::ALLOC_HEIGHT);

				#pragma unroll
				for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {

					int4 offset_x;
					int4 offset_y;

					float4 w;

					offset_x.x = offset_x_batch[s][f].x;
					offset_x.y = offset_x_batch[s][f].y;
					offset_x.z = offset_x_batch[s][f].z;
					offset_x.w = offset_x_batch[s][f].w;

					w.x = weights_batch[s][f].x;
					w.y = weights_batch[s][f].y;
					w.z = weights_batch[s][f].z;
					w.w = weights_batch[s][f].w;

/*
					offset_x.x = offset_x_batch_sh[s][f].x;
					offset_x.y = offset_x_batch_sh[s][f].y;
					offset_x.z = offset_x_batch_sh[s][f].z;
					offset_x.w = offset_x_batch_sh[s][f].w;

					w.x = weights_batch_sh[s][f].x;
					w.y = weights_batch_sh[s][f].y;
					w.z = weights_batch_sh[s][f].z;
					w.w = weights_batch_sh[s][f].w;
*/

					//for (int g; g < G; ++g)
					int g = 0;
					{
						float4 value;
						int4 sh_image_offset;
						sh_image_offset.x = offset_x.x;
						sh_image_offset.y = offset_x.y;
						sh_image_offset.z = offset_x.z;
						sh_image_offset.w = offset_x.w;

						for (int px = 0; px < BATCH_PIXELS_SIZE/4; ++px) {
							// use void* pointer to avoid doing another multiplation by sizeof(float) and instead embeed this value directly into offsets
							float4 tmp;
							float4 tmp1 = ((float4*)(s_image_sh + sh_image_offset.x + (0 + px * BATCH_PIXELS_SIZE) * sizeof(float)))[0];
							out_val[f*4 + 0][px].x +=  w.x * tmp1.x;
							out_val[f*4 + 0][px].y +=  w.x * tmp1.y;
							out_val[f*4 + 0][px].z +=  w.x * tmp1.z;
							out_val[f*4 + 0][px].w +=  w.x * tmp1.w;

							float4 tmp2 = ((float4*)(s_image_sh + sh_image_offset.y + (0 + px * BATCH_PIXELS_SIZE) * sizeof(float)))[0];
							out_val[f*4 + 1][px].x +=  w.y * tmp2.x;
							out_val[f*4 + 1][px].y +=  w.y * tmp2.y;
							out_val[f*4 + 1][px].z +=  w.y * tmp2.z;
							out_val[f*4 + 1][px].w +=  w.y * tmp2.w;

							float4 tmp3 = ((float4*)(s_image_sh + sh_image_offset.z + (0 + px * BATCH_PIXELS_SIZE) * sizeof(float)))[0];
							out_val[f*4 + 2][px].x +=  w.z * tmp3.x;
							out_val[f*4 + 2][px].y +=  w.z * tmp3.y;
							out_val[f*4 + 2][px].z +=  w.z * tmp3.z;
							out_val[f*4 + 2][px].w +=  w.z * tmp3.w;

							float4 tmp4 = ((float4*)(s_image_sh + sh_image_offset.w + (0 + px * BATCH_PIXELS_SIZE) * sizeof(float)))[0];
							out_val[f*4 + 3][px].x +=  w.w * tmp4.x;
							out_val[f*4 + 3][px].y +=  w.w * tmp4.y;
							out_val[f*4 + 3][px].z +=  w.w * tmp4.z;
							out_val[f*4 + 3][px].w +=  w.w * tmp4.w;

						}
					}
				}
			}

			__syncthreads();
		}
	}

	// TODO: we can perform shuffle between output registers and ensure only coalesed output using only STG.128
	#pragma unroll
	for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
		#pragma unroll
		for (int px = 0; px < BATCH_PIXELS_SIZE; px+=4) {

			output_batch4[OFFSET(n, f + f_offset, (block_y + thread_y ), (block_x + thread_x + px), I, F, img_height, img_width)/4] = out_val[f][px/4];
		}
	}

#endif
}

template <int Bx, int By, int BATCH_PIXELS_SIZE, int BATCH_FEATURES_SIZE, int BATCH_SUBFEATURES_SIZE, int MAX_OFFSET>
__global__ void
fast_gauss_forward_shuffle_kernel(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_weights, float* output,
							const int I, const int S, const int F, const int G,
							const int img_width, const int img_height,
							const int kernel_width, const int kernel_height) {

// INPUT: filtered images  	[I x S x H x W]
//		  filter offsets   	[F x S x G]
//		  filter weights   	[F x S x G]
// OUTPUT output  		 	[I x F x H x W]

#define BATCH_PIXELS_SIZE_SHUFFLE 4

#ifndef CUBIN_EMBEDDING
//#define MAX_OFFSET 4
/*
#define IMAGE_SH_PIXELS_X (Bx*BATCH_PIXELS_SIZE_SHUFFLE + 2*MAX_OFFSET)
#define IMAGE_SH_PIXELS_Y (By + 2*MAX_OFFSET)

#define IMAGE_SH_ARRAY_HEIGHT IMAGE_SH_PIXELS_Y

// we can load 4 pixels with one LOAD operation so arrray needs to buffer be a multiple of 4
// also make sure to add LOAD for last few pixels
#define IMAGE_SH_ARRAY_WIDTH ((IMAGE_SH_PIXELS_X + BATCH_PIXELS_SIZE_SHUFFLE-1)/BATCH_PIXELS_SIZE_SHUFFLE)
// IMAGE_SH_PITCHED_WIDTH == actualy width of image_sh_align in floats
#define IMAGE_SH_PITCHED_WIDTH (IMAGE_SH_ARRAY_WIDTH)*BATCH_PIXELS_SIZE_SHUFFLE

#if BATCH_PIXELS_SIZE % 4 == 0
	__shared__ float4 image_sh_align[BATCH_SUBFEATURES_SIZE][IMAGE_SH_ARRAY_HEIGHT][IMAGE_SH_ARRAY_WIDTH];

	float4* output_batch = reinterpret_cast<float4*>(output);
#elif BATCH_PIXELS_SIZE % 2 == 0
	__shared__ float2 image_sh_align[BATCH_SUBFEATURES_SIZE][IMAGE_SH_ARRAY_HEIGHT][IMAGE_SH_ARRAY_WIDTH];

	float2* output_batch = reinterpret_cast<float2*>(output);
#else
	__shared__ float image_sh_align[BATCH_SUBFEATURES_SIZE][IMAGE_SH_ARRAY_HEIGHT][IMAGE_SH_ARRAY_WIDTH];

	float* output_batch = reinterpret_cast<float*>(output);
#endif
*/
	static const int NUM_THREADS = Bx * By;
	static const int BATCH_SH_PIXELS_SIZE = 4;


	int off1 = 0; //offset_x_batch_sh[0][0];
	int off2 = 0; //offset_x_batch_sh[1][2];

	int block_x = BATCH_PIXELS_SIZE * (blockIdx.x * Bx);
	int block_y = blockIdx.y * By;

	int thread_x = BATCH_PIXELS_SIZE * threadIdx.x;
	int thread_y = threadIdx.y;
	int n = blockIdx.z;

	typedef BlockSharedMemory<NUM_THREADS, Bx, By * BATCH_PIXELS_SIZE, MAX_OFFSET, BATCH_SUBFEATURES_SIZE, float4, BATCH_SH_PIXELS_SIZE> SharedMem;

	__shared__ typename SharedMem::Data data;

	SharedMem image_sh_class(data, make_int2(thread_x, thread_y));

	float4* output_batch = reinterpret_cast<float4*>(output);

	float* image_sh = image_sh_class.getData(0);

	__shared__ int offset_x_batch_sh[BATCH_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE];
	__shared__ int offset_y_batch_sh[BATCH_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE];
	__shared__ float weights_batch_sh[BATCH_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE];

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

		#pragma unroll
		for (int s = 0; s < BATCH_SUBFEATURES_SIZE; ++s) {

			#pragma unroll
			for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
				{
					offset_x_batch_sh[s][f] = 0;
					offset_y_batch_sh[s][f]= 0;

					weights_batch_sh[s][f] = 1;
				}
			}
		}
	}

	__syncthreads();


	// position pointer for image loaded in hared memory at actual (x,y) position for this thread
	// (also take into account MAX_OFFSET appron added at borders)
//	float* image_sh_at_thread = image_sh + (MAX_OFFSET + thread_y) * SharedMem::PITCHED_WIDTH + MAX_OFFSET + thread_x;
	float* image_sh_at_thread = image_sh + (MAX_OFFSET + off1) * SharedMem::PITCHED_WIDTH + MAX_OFFSET + off2;

	//for (int i = 0; i < I; ++i) {
	{

		// TODO: set BATCH_PIXELS_SIZE to 1 and load N subfeatures at once into reg instead

		for (int f_offset = 0; f_offset < F; f_offset+=BATCH_FEATURES_SIZE) {

			float4 out_val[BATCH_FEATURES_SIZE];

			for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {

				out_val[f].x = 0;
				out_val[f].y = 0;
				out_val[f].z = 0;
				out_val[f].w = 0;

			}
			for (int s_offset = 0; s_offset < S; s_offset+=BATCH_SUBFEATURES_SIZE) {


				int offset_x_batch[BATCH_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE];
				int offset_y_batch[BATCH_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE];
				float weights_batch[BATCH_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE];

				float4 subfeature_batch[BATCH_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE];

				//int start_j = threadIdx.y - MAX_OFFSET;
				int start_j = -MAX_OFFSET;
				int end_j = By + MAX_OFFSET;
				//int start_i = threadIdx.x*BATCH_PIXELS_SIZE - MAX_OFFSET;
				int start_i = -MAX_OFFSET;
				int end_i = Bx*BATCH_PIXELS_SIZE + MAX_OFFSET;

				for (int s = 0; s < BATCH_SUBFEATURES_SIZE; ++s) {

					#pragma unroll
					for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {

						//for (int g; g < G; ++g)
						int g = 0;
						{
							// we can directly use offsets even with negative values since pointer is centered at (0,0)
							const int param_offset = OFFSET(0,s + s_offset,g,f + f_offset,1,S,G,F);

							//offset_x_batch[s][f] = filter_offsets_x[param_offset];
							//offset_y_batch[s][f]= filter_offsets_y[param_offset];

							//weights_batch[s][f] = filter_weights[param_offset];
							offset_x_batch[s][f] = offset_x_batch_sh[s][f];
							offset_y_batch[s][f]= offset_y_batch_sh[s][f];

							weights_batch[s][f] = weights_batch_sh[s][f];
						}
					}

					float4* s_image_sh_batch = reinterpret_cast<float4*>(image_sh + s*(SharedMem::PITCHED_WIDTH*SharedMem::ALLOC_HEIGHT));
					const float* s_filtered_image = filtered_images + OFFSET(n, s_offset + s, 0, 0, I, S, img_height, img_width);

					#pragma unroll
					for (int j = start_j; j < end_j; j+=By) {
						#pragma unroll
						for (int i = start_i; i < end_i; i+=Bx*BATCH_PIXELS_SIZE) {
							// current_image already at position for this block

							if (i + thread_x <  Bx*BATCH_PIXELS_SIZE + MAX_OFFSET && j + thread_y < By + MAX_OFFSET) {

							float4 tmp;
							tmp.x = IS_VALID_PIXEL(i + 0 + block_x + thread_x, j + block_y + thread_y, img_width, img_height) ? s_filtered_image[(j + block_y + thread_y) * img_width + block_x + thread_x + i + 0] : 0;
							tmp.y = IS_VALID_PIXEL(i + 1 + block_x + thread_x, j + block_y + thread_y, img_width, img_height) ? s_filtered_image[(j + block_y + thread_y) * img_width + block_x + thread_x + i + 1] : 0;
							tmp.z = IS_VALID_PIXEL(i + 2 + block_x + thread_x, j + block_y + thread_y, img_width, img_height) ? s_filtered_image[(j + block_y + thread_y) * img_width + block_x + thread_x + i + 2] : 0;
							tmp.w = IS_VALID_PIXEL(i + 3 + block_x + thread_x, j + block_y + thread_y, img_width, img_height) ? s_filtered_image[(j + block_y + thread_y) * img_width + block_x + thread_x + i + 3] : 0;
							tmp.x = 1;
							tmp.y = 1;
							tmp.z = 1;
							tmp.w = 1;

							if (threadIdx.z > 1000)
								s_image_sh_batch[(j + thread_y + MAX_OFFSET)*SharedMem::ALLOC_WIDTH + (i + thread_x + MAX_OFFSET)/BATCH_PIXELS_SIZE] = tmp;
							}
						}
					}

					// position subfeature pointer to start in shared memory
					const float* s_image_sh = image_sh_at_thread + s*(SharedMem::PITCHED_WIDTH*SharedMem::ALLOC_HEIGHT);

					#pragma unroll
					for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {

						//for (int g; g < G; ++g)
						int g = 0;
						{

							int sh_image_offset = offset_x_batch[s][f];

							subfeature_batch[s][f].x = s_image_sh[sh_image_offset + 0];
							subfeature_batch[s][f].y = s_image_sh[sh_image_offset + 1];
							subfeature_batch[s][f].z = s_image_sh[sh_image_offset + 2];
							subfeature_batch[s][f].w = s_image_sh[sh_image_offset + 3];
						}
					}

				}

				// make sure all threads have finished loading memory
				//__syncthreads();

#if __CUDA_ARCH__ >= 200
//  asm("bar.arrive 15, 1536;"); // # of threads must be greater than 0
#endif

				#pragma unroll
				for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {

					#pragma unroll
					for (int s = 0; s < BATCH_SUBFEATURES_SIZE ; ++s) {

						// position subfeature pointer to start in shared memory
						const float* s_image_sh = image_sh_at_thread + s*(SharedMem::PITCHED_WIDTH*SharedMem::ALLOC_HEIGHT);

						//for (int g; g < G; ++g)
						int g = 0;
						{
							// we can directly use offsets even with negative values since pointer is centered at (0,0)
							const int param_offset = OFFSET(0,s + s_offset,g,f + f_offset,1,S,G,F);

//							printf("read offset from s (%d + %d) and f (%d + %d) to get %d\n", s, s_offset, f , f_offset, OFFSET(0,s + s_offset,g,f + f_offset,1,S,G,F) );

							//int offset_x = filter_offsets_x[param_offset];
							//int offset_y = filter_offsets_y[param_offset];

							//float w = filter_weights[param_offset];

							int offset_x = offset_x_batch[s][f];
							int offset_y = offset_x_batch[s][f];

							float w = weights_batch[s][f];

							//float w = f*s/(BATCH_FEATURES_SIZE*BATCH_SUBFEATURES_SIZE);
							//int offset_x = f/(2*MAX_OFFSET) - MAX_OFFSET;
							//int offset_y = f/(2*MAX_OFFSET) - MAX_OFFSET;

							float4 value;
//							printf("s_image_sh offset %d from offsets (%d,%d) and threads (%d,%d)\n", (thread_y + offset_y) * SharedMem::PITCHED_WIDTH + thread_x + 0 + offset_x, offset_x,offset_y, thread_x, thread_y);

							//int sh_image_offset = (offset_y) * SharedMem::PITCHED_WIDTH + offset_x;
							int sh_image_offset = offset_x;

							out_val[f].x += w * subfeature_batch[s][f].x;
							out_val[f].y += w * subfeature_batch[s][f].y;
							out_val[f].z += w * subfeature_batch[s][f].z;
							out_val[f].w += w * subfeature_batch[s][f].w;

							/*out_val[f].x += w * s_image_sh[sh_image_offset + 0];
							out_val[f].y += w * s_image_sh[sh_image_offset + 1];
							out_val[f].z += w * s_image_sh[sh_image_offset + 2];
							out_val[f].w += w * s_image_sh[sh_image_offset + 3];*/

							//out_val[f].x += w * (0.28f * s_image_sh[sh_image_offset + 0] + 0.22f * s_image_sh[sh_image_offset + 1] + 0.23f * s_image_sh[sh_image_offset + SharedMem::PITCHED_WIDTH + 0] + 0.27f * s_image_sh[sh_image_offset + SharedMem::PITCHED_WIDTH + 1]);
							//out_val[f].y += w * (0.28f * s_image_sh[sh_image_offset + 1] + 0.22f * s_image_sh[sh_image_offset + 2] + 0.23f * s_image_sh[sh_image_offset + SharedMem::PITCHED_WIDTH + 1] + 0.27f * s_image_sh[sh_image_offset + SharedMem::PITCHED_WIDTH + 2]);
							//out_val[f].z += w * (0.28f * s_image_sh[sh_image_offset + 2] + 0.22f * s_image_sh[sh_image_offset + 3] + 0.23f * s_image_sh[sh_image_offset + SharedMem::PITCHED_WIDTH + 2] + 0.27f * s_image_sh[sh_image_offset + SharedMem::PITCHED_WIDTH + 3]);
							//out_val[f].w += w * (0.28f * s_image_sh[sh_image_offset + 3] + 0.22f * s_image_sh[sh_image_offset + 4] + 0.23f * s_image_sh[sh_image_offset + SharedMem::PITCHED_WIDTH + 3] + 0.27f * s_image_sh[sh_image_offset + SharedMem::PITCHED_WIDTH + 4]);
						}

					}
				}
			}
			for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
				output_batch[OFFSET(n, f + f_offset, block_y + thread_y, (block_x + thread_x)/BATCH_PIXELS_SIZE, I, F, img_height, img_width/BATCH_PIXELS_SIZE)] = out_val[f];
			}
		}
	}

#endif
}


struct  __builtin_align__(16) off4
{
	float quad[4];
};

__device__
uint get_smid(void) {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;

}


template <int BATCH_PIXELS_SIZE_X,
		int BATCH_PIXELS_SIZE_Y,
		int BATCH_FEATURES_SIZE,
		int BATCH_COMPUTE_FEATURES_SIZE,
		int BATCH_COMPUTE_SUBFEATURES_SIZE,
		int BATCH_MEM_SUBFEATURES_SIZE,
		int IMG_WIDTH, int IMG_HEIGHT>
class PipelineTextureEngine {

public:

	__device__
	PipelineTextureEngine() {

	}

	// load offset
	struct {
		bool enabled;
		float4* offset_address;
		int base_offset;
		off4* output;
	} load_offset_x, load_offset_y;

	// load w
	struct {
		bool enabled;
		float4* address;
		float4* output;
	} load_weights;

	// load data
	struct {
		bool enabled;
		off4* address_x;
		off4* address_y;
		cudaTextureObject_t image_tex;
		int tex_layer;
		float4* output;	// [BATCH_F][BATCH_PIXELS/4]
	} load_data;

	// compute
	struct {
		bool enabled;
		float4* weights;
		float4* data;
		float4* output; // [BATCH_F][BATCH_PIXELS/4]
	} compute;

	// block
	int block_x;
	int block_y;

	int current_feature;
	int current_image_id;

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

		// load quad of w for next one
		if (load_weights.enabled){
			// p.next_w = *(float4)p.next_w_address
			for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/4; ++f_quad_index )
				COPY_VECTOR4(load_weights.output[f_quad_index],load_weights.address[f_quad_index]); // weights for F[0], F[1], F[2], F[3]
		}

		static const float off_half = 0;

		// load quad of offsets for next one and make it directly into pointer to data
		if (load_offset_x.enabled) {
			for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/4; ++f_quad_index ) {
				load_offset_x.output[f_quad_index].quad[0] = load_offset_x.offset_address[f_quad_index].x + load_offset_x.base_offset + off_half; // F[0]
				load_offset_x.output[f_quad_index].quad[1] = load_offset_x.offset_address[f_quad_index].y + load_offset_x.base_offset + off_half; // F[1]
				load_offset_x.output[f_quad_index].quad[2] = load_offset_x.offset_address[f_quad_index].z + load_offset_x.base_offset + off_half; // F[2]
				load_offset_x.output[f_quad_index].quad[3] = load_offset_x.offset_address[f_quad_index].w + load_offset_x.base_offset + off_half; // F[3]
			}
		}
		if (load_offset_y.enabled) {
			for (int f_quad_index = 0; f_quad_index < BATCH_COMPUTE_FEATURES_SIZE/4; ++f_quad_index ) {
				load_offset_y.output[f_quad_index].quad[0] = load_offset_y.offset_address[f_quad_index].x + load_offset_y.base_offset + off_half; // F[0]
				load_offset_y.output[f_quad_index].quad[1] = load_offset_y.offset_address[f_quad_index].y + load_offset_y.base_offset + off_half; // F[1]
				load_offset_y.output[f_quad_index].quad[2] = load_offset_y.offset_address[f_quad_index].z + load_offset_y.base_offset + off_half; // F[2]
				load_offset_y.output[f_quad_index].quad[3] = load_offset_y.offset_address[f_quad_index].w + load_offset_y.base_offset + off_half; // F[3]
			}
		}


		#pragma unroll
		for (int i = 0; i < BATCH_COMPUTE_FEATURES_SIZE * (BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4; ++i) {

			// i goes over [BATCH_FEATURES_SIZE][BATCH_PIXELS_SIZE_/4] array so get indexes for both manually
			int f = i / ((BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4);
			int px = i % ((BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4);

			// since we store weight and offset into float4/int4 we need a proper index to access array of quad vectors
			int f_quad_index = f/4;

			// decide wheater to use float4 by width or by height:
			// if BATCH_PIXELS_SIZE_X can take at least float4 then make it by width otherwise make it by height
			static const bool BATCH_PIXELS_BY_WIDTH = BATCH_PIXELS_SIZE_X >= 4 ? true : false;

			// load data for next loop
			if (load_data.enabled) {

				if (BATCH_PIXELS_BY_WIDTH) {
					load_data.output[i] = tex2DLayered<float4>(load_data.image_tex, load_data.address_x[f_quad_index].quad[f % 4], load_data.address_y[f_quad_index].quad[f % 4], load_data.tex_layer);
				} else {
					assert("BATCH_PIXELS_SIZE_X must be >=4 when using texture");
				}
			}

			// compute for current loop
			if (compute.enabled) {
				//if(threadIdx.x == 0)
				//	printf("computing for i: %d, f: %d, px: %d\n",i,f,px);
				compute.output[i].x += compute.weights[f_quad_index].x * compute.data[i].x;
				compute.output[i].y += compute.weights[f_quad_index].y * compute.data[i].y;
				compute.output[i].z += compute.weights[f_quad_index].z * compute.data[i].z;
				compute.output[i].w += compute.weights[f_quad_index].w * compute.data[i].w;
			}
		}

		//__threadfence_block();
	}

};




__device__ inline
void compute_dot_vals(float4& out_compute_val, float4 & in_compute_val, float w) {
	out_compute_val.x += w * in_compute_val.x;
	out_compute_val.y += w * in_compute_val.y;
	out_compute_val.z += w * in_compute_val.z;
	out_compute_val.w += w * in_compute_val.w;
}



__device__ inline
float4 load_texture(cudaTextureObject_t& image_tex, float tex_offset_x, float tex_offset_y, int tex_layer) {
	float4 value;

	//value = tex2DLayered<float4>(image_tex, tex_offset_x + 0.5, tex_offset_y + 0.5, tex_layer);
	return tex2DLayered<float4>(image_tex, tex_offset_x, tex_offset_y, tex_layer);
}


template <typename BlockIndexingT>
__global__ void
fast_gauss_forward_textrue_kernel(cudaTextureObject_t* filtered_images_tex, const float* filtered_images, const float* filter_offsets_x, const float* filter_offsets_y, const float* filter_weights, float* output,
							const int I, const int S, const int F, const int G,
							const int img_width_, const int img_height_,
							const int kernel_width, const int kernel_height) {

// INPUT: filtered images  	[I x S x H x W]
//		  filter offsets   	[F x S x G]
//		  filter weights   	[F x S x G]
// OUTPUT output  		 	[I x F x H x W]

#ifndef CUBIN_EMBEDDING

	typedef class BlockIndexingT::Kernel BlockIndexingKernel;

	static const int NUM_SM = BlockIndexingT::NUM_SM;
	static const int Bx = BlockIndexingT::Bx;
	static const int By = BlockIndexingT::By;
	static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
	static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
	static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
	static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
	static const int BATCH_COMPUTE_SUBFEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE;
	static const int BATCH_MEM_SUBFEATURES_SIZE = BlockIndexingT::BATCH_MEM_SUBFEATURES_SIZE;
	static const int IMG_WIDTH = BlockIndexingT::IMG_WIDTH;
	static const int IMG_HEIGHT = BlockIndexingT::IMG_HEIGHT;
	static const int MAX_OFFSET = BlockIndexingT::MAX_OFFSET;


	int img_width = IMG_WIDTH;
	int img_height = IMG_HEIGHT;

	///const float4* filtered_images4 = reinterpret_cast<const float4*>(filtered_images);
	float4* output4 = reinterpret_cast<float4*>(output);

	BlockIndexingKernel block_indexing(img_width, img_height);

	int i = block_indexing.getImageIdx();

	int f_offset = block_indexing.getFeatureIdx();


	int block_width = block_indexing.getPosBlockSize().x;
	int block_height = block_indexing.getPosBlockSize().y;

	int block_x = block_indexing.getPosBlockIdx().x;
	int block_y = block_indexing.getPosBlockIdx().y;

	int thread_x = block_indexing.getPosThreadIdx().x;
	int thread_y = block_indexing.getPosThreadIdx().y;


	thread_x  += block_x;
	thread_y  += block_y;

	cudaTextureObject_t current_image_tex = filtered_images_tex[i];

	/*
	int blk_id = blockIdx.x  +
				(gridDim.x) * blockIdx.y +
				(gridDim.x * gridDim.y) * blockIdx.z;


	__threadfence_block();
	if (threadIdx.x == 0 || true) {
		clock_t timestamp = clock();
		//printf("time: %llu, started kernel at SM: %d, image: %d, feature %d, block: %d, %d: block idx: %d\n", timestamp,  (int)get_smid(), i,  f_offset, block_x, block_y, blk_id);
		//printf("time: %llu, started kernel at SM: %d, image: %d, feature %d, block: %d, %d: block idx: %d, block x: %d, y: %d, z:%d dim: %d,%d,%d \n", timestamp,  (int)get_smid(),  blockIdx.z,  f_offset, block_x, block_y, blk_id, blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z);
	}

	__threadfence_block();
	*/
/*
	float4 out_val[BATCH_FEATURES_SIZE];

	for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
		out_val[f].x = 0;
		out_val[f].y = 0;
		out_val[f].z = 0;
		out_val[f].w = 0;
	}

	static const int const BATCH_SUBFEATURES_SIZE = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE;


	for (int s_offset = 0; s_offset < S; s_offset+=BATCH_SUBFEATURES_SIZE) {

		const int* filter_offsets_x_s_off = filter_offsets_x + (s_offset) * (BATCH_FEATURES_SIZE) + f_offset;
		const int* filter_offsets_y_s_off = filter_offsets_y + (s_offset) * (BATCH_FEATURES_SIZE) + f_offset;
		const float* filter_weights_s_off = filter_weights + (s_offset) * (BATCH_FEATURES_SIZE) + f_offset;

		#pragma unroll
		for (int s = 0; s < BATCH_SUBFEATURES_SIZE; ++s ){

			#pragma unroll
			for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {

				// load offset and weights for doubled buffering
				float4 w = reinterpret_cast<const float4*>(filter_weights_s_off)[(s) * (BATCH_FEATURES_SIZE/4) + f];

				int4 offset_x = reinterpret_cast<const int4*>(filter_offsets_x_s_off)[(s) * (BATCH_FEATURES_SIZE/4) + f];
				int4 offset_y = reinterpret_cast<const int4*>(filter_offsets_y_s_off)[(s) * (BATCH_FEATURES_SIZE/4) + f];

				//for (int g = 0; g < G; ++g)
				int g = 0;
				{
					//int offset_x = filter_offsets_x[OFFSET(0,s,g,f_offset+f,1,S,G,F)];
					//int offset_y = filter_offsets_y[OFFSET(0,s,g,f_offset+f,1,S,G,F)];


					//float w = filter_weights[OFFSET(0,s,g,f_offset+f,1,S,G,F)];

					//int offset_x = f/(2*MAX_OFFSET) - MAX_OFFSET;
					//int offset_y = f/(2*MAX_OFFSET) - MAX_OFFSET;

					//float w = 1;

					float4 value;

					//value = tex2DLayered<float4>(current_image_tex, (thread_x + offset_x.x) / BATCH_PIXELS_SIZE + 0.5f, offset_y.x + thread_y + 0.5f, s_offset + s);
					value = tex2DLayered<float4>(current_image_tex, (thread_x + offset_x.x) / BATCH_PIXELS_SIZE, offset_y.x + thread_y, s_offset + s);
					out_val[f*4 + 0].x += w.x * value.x;
					out_val[f*4 + 0].y += w.x * value.y;
					out_val[f*4 + 0].z += w.x * value.z;
					out_val[f*4 + 0].w += w.x * value.w;

					//value = tex2DLayered<float4>(current_image_tex, (thread_x + offset_x.y) / BATCH_PIXELS_SIZE + 0.5f, offset_y.y + thread_y + 0.5f, s_offset + s);
					value = tex2DLayered<float4>(current_image_tex, (thread_x + offset_x.y) / BATCH_PIXELS_SIZE , offset_y.y + thread_y, s_offset + s);
					out_val[f*4 + 1].x += w.y * value.x;
					out_val[f*4 + 1].y += w.y * value.y;
					out_val[f*4 + 1].z += w.y * value.z;
					out_val[f*4 + 1].w += w.y * value.w;

					//value = tex2DLayered<float4>(current_image_tex, (thread_x + offset_x.z) / BATCH_PIXELS_SIZE + 0.5f, offset_y.z + thread_y + 0.5f, s_offset + s);
					value = tex2DLayered<float4>(current_image_tex, (thread_x + offset_x.z) / BATCH_PIXELS_SIZE, offset_y.z + thread_y, s_offset + s);
					out_val[f*4 + 2].x += w.z * value.x;
					out_val[f*4 + 2].y += w.z * value.y;
					out_val[f*4 + 2].z += w.z * value.z;
					out_val[f*4 + 2].w += w.z * value.w;

					//value = tex2DLayered<float4>(current_image_tex, (thread_x + offset_x.w) / BATCH_PIXELS_SIZE + 0.5f, offset_y.w + thread_y + 0.5f, s_offset + s);
					value = tex2DLayered<float4>(current_image_tex, (thread_x + offset_x.w) / BATCH_PIXELS_SIZE, offset_y.w + thread_y , s_offset + s);
					out_val[f*4 + 3].x += w.w * value.x;
					out_val[f*4 + 3].y += w.w * value.y;
					out_val[f*4 + 3].z += w.w * value.z;
					out_val[f*4 + 3].w += w.w * value.w;

					//printf("written i,f,y,x: (%d,%d,%d,%d) with offset %d and values %f,%f,%f,%f\n", i, f, thread_y, thread_x + 0, OFFSET(i, f, thread_y, thread_x + 0, I, F, img_height, img_width), value.x, value.y, value.z, value.w);
				}
			}
		}
	}

//printf("written i,f,y,x: (%d,%d,%d,%d) with offset %d and values %f,%f,%f,%f\n", i, f, thread_y, thread_x + 0, OFFSET(i, f, thread_y, thread_x + 0, I, F, img_height, img_width), out_val.x, out_val.y, out_val.z, out_val.w);

	//output4[OFFSET(i, f, thread_y, thread_x/4, I, F, img_height, img_width/4)] = out_val;

	for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
		output4[OFFSET(i, f_offset+f, thread_y, thread_x, I, F, img_height, img_width)/BATCH_PIXELS_SIZE] = out_val[f];
	}

*/

	__shared__ float4 offset_x_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];
	__shared__ float4 offset_y_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];
	__shared__ float4 weights_batch_sh[BATCH_MEM_SUBFEATURES_SIZE][BATCH_FEATURES_SIZE/4];

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

		#pragma unroll
		for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

			#pragma unroll
			for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {
				{
					offset_x_batch_sh[s][f].x = 0;
					offset_x_batch_sh[s][f].y = 0;
					offset_x_batch_sh[s][f].z = 0;
					offset_x_batch_sh[s][f].w = 0;

					offset_y_batch_sh[s][f].x = 0;
					offset_y_batch_sh[s][f].y = 0;
					offset_y_batch_sh[s][f].z = 0;
					offset_y_batch_sh[s][f].w = 0;

					weights_batch_sh[s][f].x = 1;
					weights_batch_sh[s][f].y = 1;
					weights_batch_sh[s][f].z = 1;
					weights_batch_sh[s][f].w = 1;
				}
			}
		}
	}

	__syncthreads();

	float4 out_val[BATCH_FEATURES_SIZE][(BATCH_PIXELS_SIZE_X*BATCH_PIXELS_SIZE_Y)/4];

	for (int f = 0; f < BATCH_FEATURES_SIZE ; ++f) {
		for (int px = 0; px < (BATCH_PIXELS_SIZE_X*BATCH_PIXELS_SIZE_Y)/4; ++px) {

			out_val[f][px].x = 0;
			out_val[f][px].y = 0;
			out_val[f][px].z = 0;
			out_val[f][px].w = 0;
		}
	}


	static const int BATCH_COMPUTE_FEATURES_SIZE = 4;

	// batch pixels by width
	PipelineTextureEngine<BATCH_PIXELS_SIZE_X,
					BATCH_PIXELS_SIZE_Y,
					BATCH_FEATURES_SIZE,
					BATCH_COMPUTE_FEATURES_SIZE,
					BATCH_COMPUTE_SUBFEATURES_SIZE,
					BATCH_MEM_SUBFEATURES_SIZE,
					IMG_WIDTH, IMG_HEIGHT> pipeline;


	pipeline.current_image_id = i;
	pipeline.block_x = block_x;
	pipeline.block_y = block_y;

	float thread_offset_x = thread_x/4;
	float thread_offset_y = thread_y/4;

	for (int s_offset_outer = 0; s_offset_outer < S; s_offset_outer+=BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE) {

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		{

			#pragma unroll
			for (int s = 0; s < BATCH_MEM_SUBFEATURES_SIZE; ++s) {

				#pragma unroll
				for (int f = 0; f < BATCH_FEATURES_SIZE/4; ++f) {
				//int f = 0; {
					//  TODO: use multiple threads to load values effectivly only one instruction to load offset and weigts !!
					offset_x_batch_sh[s][f] = reinterpret_cast<const float4*>(filter_offsets_x)[(s_offset_outer + s) * (BATCH_FEATURES_SIZE/4) + f];
					offset_y_batch_sh[s][f] = reinterpret_cast<const float4*>(filter_offsets_y)[(s_offset_outer + s) * (BATCH_FEATURES_SIZE/4) + f];
					weights_batch_sh[s][f] = reinterpret_cast<const float4*>(filter_weights)[(s_offset_outer + s) * (BATCH_FEATURES_SIZE/4) + f];
				}
			}
		}


		off4 off_x_A[BATCH_COMPUTE_FEATURES_SIZE/4],
			 off_x_B[BATCH_COMPUTE_FEATURES_SIZE/4];

		off4 off_y_A[BATCH_COMPUTE_FEATURES_SIZE/4],
			 off_y_B[BATCH_COMPUTE_FEATURES_SIZE/4];

		float4 w_A[BATCH_COMPUTE_FEATURES_SIZE/4],
			   w_B[BATCH_COMPUTE_FEATURES_SIZE/4];

		float4 d_A[BATCH_FEATURES_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4],
			   d_B[BATCH_FEATURES_SIZE][(BATCH_PIXELS_SIZE_X * BATCH_PIXELS_SIZE_Y)/4];

		//const float* current_image_global = filtered_images + OFFSET(n, s_offset_outer, block_y + image_sh_class.getThreadIdx().y, block_x + image_sh_class.getThreadIdx().x, I, S, img_height, img_width);

		struct IterIndex {
			int s; // sub-feature
			int f; // feature
		};

		int NUM_ITER = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE;

		// do all in one loop
		#pragma unroll
		for (int index = 0 ; index < NUM_ITER + 2; ++index)  {

			int s = index / (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE);
			int f = (index % (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE)) * BATCH_COMPUTE_FEATURES_SIZE;

/*
			__threadfence_block();
			if (threadIdx.x == 0) {
				clock_t timestamp = clock();
			//	printf("time: %llu, SM: %d, loading image: %d, feature: %d, subfeature: %d, block: %d, %d, actual block id: %d, %d, %d\n", timestamp, (int)get_smid(), i, f + f_offset, s + s_offset_outer, block_x, block_y, blockIdx.x,blockIdx.y,blockIdx.z);
			}
			__threadfence_block();
*/
/*
			if (f == 0) {
				if (threadIdx.z == 1023022) {
					weights_batch_sh[0][5].x = 10;
				}
			}
*/
			// get s and f index values for this loop
			IterIndex load_global;
			IterIndex load_offset_index;
			IterIndex load_w_index;
			IterIndex load_data_index;
			IterIndex compute_index;

			// global loading is done imediately (no delay)
			const int start_delay_offset_load = 0;
			const int start_delay_w_load = 1;
			const int start_delay_data_load = 1;
			const int start_delay_compute = 2;

			// get flags to run each unit based on index number and its delay
			pipeline.load_offset_x.enabled = pipeline.should_run(index, start_delay_offset_load, NUM_ITER);
			pipeline.load_offset_y.enabled = pipeline.should_run(index, start_delay_offset_load, NUM_ITER);
			pipeline.load_weights.enabled = pipeline.should_run(index, start_delay_w_load, NUM_ITER);
			pipeline.load_data.enabled = pipeline.should_run(index, start_delay_data_load, NUM_ITER);
			pipeline.compute.enabled = pipeline.should_run(index, start_delay_compute, NUM_ITER);

			{
				// offset loading is done with no delay
				load_offset_index.s = (index - start_delay_offset_load) / (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE);
				load_offset_index.f = ((index - start_delay_offset_load) % (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE)) * BATCH_COMPUTE_FEATURES_SIZE;

				int double_buffer_index = (load_offset_index.s/BATCH_MEM_SUBFEATURES_SIZE) % 2;
				int subfeat_buffer_index = load_offset_index.s % BATCH_MEM_SUBFEATURES_SIZE;

				// switch between registers every iteration
				bool use_reg_A = (index - start_delay_offset_load) % 2 == 0 ? true : false;

				// load offset
				pipeline.load_offset_x.offset_address = &offset_x_batch_sh[load_offset_index.s % BATCH_MEM_SUBFEATURES_SIZE][load_offset_index.f/4];
				pipeline.load_offset_x.base_offset = thread_offset_x;
				pipeline.load_offset_x.output = use_reg_A ? &off_x_A[0] : &off_x_B[0];

				pipeline.load_offset_y.offset_address = &offset_y_batch_sh[load_offset_index.s % BATCH_MEM_SUBFEATURES_SIZE][load_offset_index.f/4];
				pipeline.load_offset_y.base_offset = thread_offset_y;
				pipeline.load_offset_y.output = use_reg_A ? &off_y_A[0] : &off_y_B[0];
				// performs next_offset = next_offset_address[i] + next_data_address
			}

			{
				// w and data loading is done with single delay
				load_w_index.s = (index - start_delay_w_load) / (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE);
				load_w_index.f = ((index - start_delay_w_load) % (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE)) * BATCH_COMPUTE_FEATURES_SIZE;

				// switch between registers every iteration
				bool use_reg_A = (index - start_delay_w_load) % 2 == 0 ? true : false;

				// load w
				pipeline.load_weights.address = &weights_batch_sh[load_w_index.s % BATCH_MEM_SUBFEATURES_SIZE][load_w_index.f/4];
				pipeline.load_weights.output = use_reg_A ? &w_A[0] : &w_B[0];

				// performs next_w = next_w_address[i]
			}
			bool require_sync = false;
			{
				load_data_index.s = (index - start_delay_data_load) / (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE);
				load_data_index.f = ((index - start_delay_data_load) % (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE)) * BATCH_COMPUTE_FEATURES_SIZE;

				int current_double_buffer_index = (load_data_index.s/BATCH_MEM_SUBFEATURES_SIZE) % 2;
				int next_double_buffer_index = (( (index + 1 - start_delay_data_load) / (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE) ) /BATCH_MEM_SUBFEATURES_SIZE) % 2;

				require_sync = current_double_buffer_index == next_double_buffer_index ? false : true;

				// switch between registers every iteration
				bool use_reg_A = (index - start_delay_data_load) % 2 == 0 ? true : false;

				// load data

				pipeline.load_data.address_x = use_reg_A ? &off_x_A[0] : &off_x_B[0];
				pipeline.load_data.address_y = use_reg_A ? &off_y_A[0] : &off_y_B[0];
				pipeline.load_data.image_tex = current_image_tex;
				pipeline.load_data.tex_layer = s_offset_outer + load_data_index.s;
				pipeline.load_data.output = use_reg_A ? d_A[load_data_index.f] : d_B[load_data_index.f];

				pipeline.current_feature = load_data_index.f + f_offset;

				// performs next_data = next_shared_data_address[offset]
			}

			{
				// computation is done with double  delay
				compute_index.s = (index - start_delay_compute) / (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE);
				compute_index.f = ((index - start_delay_compute) % (BATCH_FEATURES_SIZE/BATCH_COMPUTE_FEATURES_SIZE)) * BATCH_COMPUTE_FEATURES_SIZE;

				// switch between registers every iteration
				bool use_reg_A = (index - start_delay_compute) % 2 == 0 ? true : false;

				// compute
				pipeline.compute.weights = use_reg_A ? w_A : w_B;
				pipeline.compute.data = use_reg_A ? d_A[compute_index.f] : d_B[compute_index.f];
				pipeline.compute.output = out_val[compute_index.f];

				// performs output[i] += weights[i] * data[i]
			}

			pipeline.execute_step();

			// sync only before data buffer is switched
//			if (require_sync)
//				__syncthreads();
		}
	}


	// TODO: we can perform shuffle between output registers and ensure only coalesed output using only STG.128
	#pragma unroll
	for (int f = 0; f < BATCH_FEATURES_SIZE; ++f) {
		if (BATCH_PIXELS_SIZE_X >= 4) {
			// version for loading per 4 pixels by width and 1 pixel per height
			#pragma unroll
			for (int px_y = 0; px_y < BATCH_PIXELS_SIZE_Y; ++px_y) {
				#pragma unroll
				for (int px_x = 0; px_x < BATCH_PIXELS_SIZE_X; px_x+=4) {
					reinterpret_cast<float4*>(output)[OFFSET(i, f + f_offset, (thread_y + px_y), (thread_x + px_x), I, F, img_height, img_width)/4] = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/4];
				}
			}
		} else {
			// version for loading per 1 pixels by width and 4 pixel per height
			#pragma unroll
			for (int px_y = 0; px_y < BATCH_PIXELS_SIZE_Y; px_y+=4) {
				#pragma unroll
				for (int px_x = 0; px_x < BATCH_PIXELS_SIZE_X; ++px_x) {
					output[OFFSET(i, f + f_offset, (thread_y + 0 + px_y), (thread_x + px_x), I, F, img_height, img_width)] = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/4].x;
					output[OFFSET(i, f + f_offset, (thread_y + 1 + px_y), (thread_x + px_x), I, F, img_height, img_width)] = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/4].y;
					output[OFFSET(i, f + f_offset, (thread_y + 2 + px_y), (thread_x + px_x), I, F, img_height, img_width)] = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/4].z;
					output[OFFSET(i, f + f_offset, (thread_y + 3 + px_y), (thread_x + px_x), I, F, img_height, img_width)] = out_val[f][(px_y * BATCH_PIXELS_SIZE_X + px_x)/4].w;
				}
			}
		}
	}

#endif
}

template <int Bx, int By, int BATCH_PIXELS_SIZE>
__global__ void
fast_gauss_forward_basic_kernel(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_weights, float* output,
							const int I, const int S, const int F, const int G,
							const int img_width, const int img_height,
							const int kernel_width, const int kernel_height) {

// INPUT: filtered images  	[I x S x H x W]
//		  filter offsets   	[F x S x G]
//		  filter weights   	[F x S x G]
// OUTPUT output  		 	[I x F x H x W]

#ifndef CUBIN_EMBEDDING


	//const float4* filtered_images4 = reinterpret_cast<const float4*>(filtered_images);
	float4* output4 = reinterpret_cast<float4*>(output);

	int thread_x = BATCH_PIXELS_SIZE * (blockIdx.x * Bx + threadIdx.x);
	int thread_y = blockIdx.y * By + threadIdx.y;
	int i = blockIdx.z;

//	for (int i = 0; i < I; ++i)
	{

		for (int f = 0; f < F; ++f) {

			float4 out_val;
			out_val.x = 0;
			out_val.y = 0;
			out_val.z = 0;
			out_val.w = 0;

			for (int s = 0; s < S; ++s) {
				for (int g = 0; g < G; ++g) {

					int offset_x = filter_offsets_x[OFFSET(0,s,g,f,1,S,G,F)];
					int offset_y = filter_offsets_y[OFFSET(0,s,g,f,1,S,G,F)];

					float w = filter_weights[OFFSET(0,s,g,f,1,S,G,F)];

					float4 value;

					value.x = IS_VALID_PIXEL(thread_x + 0 + offset_x, thread_y + offset_y, img_width, img_height) ? filtered_images[OFFSET(i, s, thread_y + offset_y, thread_x + 0 + offset_x, I, S, img_height, img_width)] : 0;
					value.y = IS_VALID_PIXEL(thread_x + 1 + offset_x, thread_y + offset_y, img_width, img_height) ? filtered_images[OFFSET(i, s, thread_y + offset_y, thread_x + 1 + offset_x, I, S, img_height, img_width)] : 0;
					value.z = IS_VALID_PIXEL(thread_x + 2 + offset_x, thread_y + offset_y, img_width, img_height) ? filtered_images[OFFSET(i, s, thread_y + offset_y, thread_x + 2 + offset_x, I, S, img_height, img_width)] : 0;
					value.w = IS_VALID_PIXEL(thread_x + 3 + offset_x, thread_y + offset_y, img_width, img_height) ? filtered_images[OFFSET(i, s, thread_y + offset_y, thread_x + 3 + offset_x, I, S, img_height, img_width)] : 0;

					//printf("i,s,y,x: (%d,%d,%d,%d) with offset %d and values %f,%f,%f,%f and weight %f\n", i, s, thread_y, thread_x + 0, OFFSET(i, s, thread_y, thread_x + 0, I, S, img_height, img_width), value.x, value.y, value.z, value.w, w);

					out_val.x += w * value.x;
					out_val.y += w * value.y;
					out_val.z += w * value.z;
					out_val.w += w * value.w;
				}
			}

			//out_val.x = S;
			//out_val.y = S;
			//out_val.z = S;
			//out_val.w = S;

			//printf("written i,f,y,x: (%d,%d,%d,%d) with offset %d and values %f,%f,%f,%f\n", i, f, thread_y, thread_x + 0, OFFSET(i, f, thread_y, thread_x + 0, I, F, img_height, img_width), out_val.x, out_val.y, out_val.z, out_val.w);

			output4[OFFSET(i, f, thread_y, thread_x/4, I, F, img_height, img_width/4)] = out_val;
			//output[OFFSET(i, f, thread_y, thread_x + 0, I, F, img_height, img_width)] = out_val.x;
			//output[OFFSET(i, f, thread_y, thread_x + 1, I, F, img_height, img_width)] = out_val.y;
			//output[OFFSET(i, f, thread_y, thread_x + 2, I, F, img_height, img_width)] = out_val.z;
			//output[OFFSET(i, f, thread_y, thread_x + 3, I, F, img_height, img_width)] = out_val.w;
		}
	}

#endif
}
#include <iostream>

#define OFFSET8(i8, i7, i6, i5, i4, i3, i2, i1, num_i8, num_i7, num_i6, num_i5, num_i4, num_i3, num_i2, num_i1) \
				((( (( ( ((i8) * (num_i7) + i7)* (num_i6)  + (i6)  )*(num_i5)  + (i5)  )   * (num_i4) + (i4))*(num_i3) + (i3)) * (num_i2) + (i2))*(num_i1) + (i1) )

template <typename BlockIndexingT>
__global__  void
perpare_weights_and_offsets(const float* filter_weights, const int* filter_offsets_x, const int* filter_offsets_y, float *prepared_filter_weights, int *prepared_filter_offsets, int S, int G, int F) {

	static const int NUM_SM = BlockIndexingT::NUM_SM;
	static const int Bx = BlockIndexingT::Bx;
	static const int By = BlockIndexingT::By;
	static const int BLOCK_FEATURES = BlockIndexingT::BLOCK_FEATURES;
	static const int BATCH_PIXELS_SIZE_X = BlockIndexingT::BATCH_PIXELS_SIZE_X;
	static const int BATCH_PIXELS_SIZE_Y = BlockIndexingT::BATCH_PIXELS_SIZE_Y;
	static const int BATCH_FEATURES_SIZE = BlockIndexingT::BATCH_FEATURES_SIZE;
	static const int BATCH_COMPUTE_SUBFEATURES_SIZE = BlockIndexingT::BATCH_COMPUTE_SUBFEATURES_SIZE;
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
	// float4 of size [F / (BATCH_FEATURES_SIZE * BLOCK_FEATURES)] x [S / (BATCH_COMPUTE_SUBFEATURES_SIZE*BATCH_MEM_SUBFEATURES_SIZE)] x [G / BATCH_GAUSS_SIZE]
	//				 	x [BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE] x [BATCH_GAUSS_SIZE] x [PIXELS_INTERPOLATION_SIZE] x [BATCH_FEATURES_SIZE/4] x [BLOCK_FEATURES];

	static const int dim1_size = BLOCK_FEATURES;
	static const int dim2_size = BATCH_FEATURES_SIZE/4;
	static const int dim3_size = 4;
	static const int dim4_size = BATCH_GAUSS_SIZE;
	static const int dim5_size = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE;

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

	if (output_offset.x % 2 == 1) output_offset.x += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
	if (output_offset.y % 2 == 1) output_offset.y += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
	if (output_offset.z % 2 == 1) output_offset.z += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;
	if (output_offset.w % 2 == 1) output_offset.w += (SharedMem::PITCHED_WIDTH * SharedMem::ALLOC_HEIGHT) - 1;

	prepared_filter_offsets4[output_index] = output_offset;

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

}


__global__ void
print2d_array(const float* input, int width, int height, int N) {

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
void fast_gauss_forward<double>(const double* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
								const double* filter_weights, double* output,
								const int I, const int S, const int F, const int G,
								const int img_width, const int img_height,
								const int kernel_width, const int kernel_height, cudaStream_t streamId) {

}

#define N 1024

// texture object is a kernel argument
__global__ void kernel(cudaTextureObject_t tex, float* output) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  float x = tex1Dfetch<float>(tex, i);

  output[i] = x*x + 2;


}



// texture object is a kernel argument
__global__ void kernel2d_tex(cudaTextureObject_t* tex, float* output) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  int j = blockIdx.y *blockDim.y + threadIdx.y;
  printf("reading data at %d,%d,0\n",i,j);
  float4 x = tex2DLayered<float4>(*tex, i + 0.5f,i+0.5f,0);

  printf("outputing mul of data at %d,%d,0 with val %f, %f, %f,%f\n",i,j, x.x, x.y, x.z, x.w);
  output[i] = x.x * x.y * x.z * x.w;


}

cudaTextureObject_t compose_float_tex_object(const float* buffer, int buffer_size)
{


	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = (void*)buffer;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = buffer_size * sizeof(float);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	// create texture object: we only have to do this once!
	cudaTextureObject_t tex=0;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	return tex;
}



float* create_input_with_border(const float* filtered_images, const int img_width, const int img_height, const int NN, const int border_size, cudaStream_t* streams, const int NUM_STREAMS ) {

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
void fast_gauss_forward<float>(const float* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
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
					((int)ceil(img_height) + threadsPerBlock.y - 1) / threadsPerBlock.y,										// over image height
					I);																								// over batch size

	std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
	std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

	cudaDeviceProp devProp;
	CUDA_CHECK(cudaGetDeviceProperties(&devProp,0));

	std::cout << "CUDA requrements: texture alignment " << devProp.textureAlignment << ", pitch alingnment " << devProp.texturePitchAlignment << std::endl;

	if (0) {
		std::cout << "started fast_gauss_forward_basic_kernel" << std::endl;

		clock_t start_t = clock();
		fast_gauss_forward_basic_kernel<BLOCK_X,BLOCK_Y,BATCH_PIXELS_SIZE><<<numBlocks,threadsPerBlock>>>(filtered_images, filter_offsets_x, filter_offsets_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
		std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
		cudaDeviceSynchronize();

		clock_t end_t = clock();
		std::cout << "fast_gauss_forward_basic_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
		//
	}


	if (0) {
		std::cout << "started fast_gauss_forward_shuffle_kernel" << std::endl;

		clock_t start_t = clock();
		fast_gauss_forward_shuffle_kernel<BLOCK_X,BLOCK_Y,BATCH_PIXELS_SIZE,BATCH_FEATURES_SIZE,BATCH_SUBFEATURES_SIZE, MAX_OFFSET><<<numBlocks,threadsPerBlock>>>(filtered_images, filter_offsets_x, filter_offsets_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
		CUDA_POST_KERNEL_CHECK;
		std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
		cudaDeviceSynchronize();

		clock_t end_t = clock();
		std::cout << "fast_gauss_forward_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
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


		if (0) {

			static const int BATCH_PIXELS_SIZE_X = 4;
			static const int BATCH_PIXELS_SIZE_Y = 1;

			static const int BLOCK_X = 16/BATCH_PIXELS_SIZE;
			static const int BLOCK_Y = 8;

			static const int BLOCK_FEATURES = 16;

			static const int BATCH_FEATURES_SIZE = 8;
			static const int BATCH_SUBFEATURES_SIZE = 1;
			static const int BATCH_COMPUTE_SUBFEATURES_SIZE = 8;
			static const int BATCH_MEM_SUBFEATURES_SIZE = 1;
			static const int BATCH_GAUSS_SIZE = 1;

			static const int MAX_OFFSET = 4;

			static const int IMG_WIDTH = 64;
			static const int IMG_HEIGHT = 64;

			static const int NUM_SM = 16; // number of streaming multiprocessors

			typedef class BlockIndexing<NUM_SM,
							BLOCK_X, BLOCK_Y, BLOCK_FEATURES,
							BATCH_PIXELS_SIZE_X, BATCH_PIXELS_SIZE_Y,
							1,1,
							BATCH_FEATURES_SIZE,
							BATCH_COMPUTE_SUBFEATURES_SIZE,
							BATCH_MEM_SUBFEATURES_SIZE,
							BATCH_GAUSS_SIZE,
							IMG_WIDTH, IMG_HEIGHT,
							MAX_OFFSET> BlockIndexingTextureT;

			BlockIndexingTextureT::Launch block_indexing;

			std::cout << "started fast_gauss_forward_textrue_kernel" << std::endl;

			dim3 threadsPerBlock_ = block_indexing.getThreadsPerBlock(I, F, S, img_width, img_height);

			dim3 numBlocks_ = block_indexing.getBlocksPerGrid(I, F, S, img_width, img_height);

			std::cout << "threadsPerBlock " << threadsPerBlock_.x << "," << threadsPerBlock_.y << "," << threadsPerBlock_.z << std::endl;
			std::cout << "numBlocks " << numBlocks_.x << "," << numBlocks_.y << "," << numBlocks_.z << std::endl;

			clock_t start_t = clock();
			fast_gauss_forward_textrue_kernel<BlockIndexingTextureT><<<numBlocks_,threadsPerBlock_>>>(filtered_images_tex_layered_gpu, filtered_images, filter_offsets_float_x, filter_offsets_float_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);

			std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
			cudaDeviceSynchronize();

			clock_t end_t = clock();
			std::cout << "fast_gauss_forward_textrue_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;
			//
		}


		if (0){

			static const int BATCH_PIXELS_SIZE_X = 1;
			static const int BATCH_PIXELS_SIZE_Y = 4;
			static const int BLOCK_FEATURES = 1;

			static const int BLOCK_X = 32/BATCH_PIXELS_SIZE_X;
			static const int BLOCK_Y = 8/BATCH_PIXELS_SIZE_Y;

			static const int BATCH_FEATURES_SIZE = 16;
			static const int BATCH_COMPUTE_SUBFEATURES_SIZE = 4;
			static const int BATCH_MEM_SUBFEATURES_SIZE = 1;
			static const int BATCH_GAUSS_SIZE = 1;

			static const int IMG_WIDTH = 64;
			static const int IMG_HEIGHT = 64;
			static const int MAX_OFFSET = 4;
			static const int NUM_SM = 1; // number of streaming multiprocessors - seems to be better with NUM_SM = 1;

			typedef class BlockIndexing<NUM_SM,
							BLOCK_X, BLOCK_Y, BLOCK_FEATURES,
							BATCH_PIXELS_SIZE_X, BATCH_PIXELS_SIZE_Y,
							1,1,
							BATCH_FEATURES_SIZE,
							BATCH_COMPUTE_SUBFEATURES_SIZE,
							BATCH_MEM_SUBFEATURES_SIZE,
							BATCH_GAUSS_SIZE,
							IMG_WIDTH, IMG_HEIGHT,
							MAX_OFFSET> BlockIndexingPipelineT;

			BlockIndexingPipelineT::Launch block_indexing;

			dim3 threadsPerBlock = block_indexing.getThreadsPerBlock(I, F, S, img_width, img_height);

			dim3 numBlocks = block_indexing.getBlocksPerGrid(I, F, S, img_width, img_height);

			std::cout << "started fast_gauss_forward_kernel" << std::endl;

			std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
			std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

			clock_t start_t = clock();
			fast_gauss_forward_kernel<BlockIndexingPipelineT><<<numBlocks,threadsPerBlock>>>(filtered_images_tex_layered_gpu, filtered_images, filter_offsets_x, filter_offsets_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
			CUDA_POST_KERNEL_CHECK;
			std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
			cudaDeviceSynchronize();

			clock_t end_t = clock();
			std::cout << "fast_gauss_forward_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

		}

		if (0){

			static const int BATCH_PIXELS_SIZE_X = 4;
			static const int BATCH_PIXELS_SIZE_Y = 1;
			static const int BLOCK_FEATURES = 1;

			static const int BLOCK_X = 32/BATCH_PIXELS_SIZE_X;
			static const int BLOCK_Y = 8/BATCH_PIXELS_SIZE_Y;

			static const int BATCH_FEATURES_SIZE = 8;
			static const int BATCH_COMPUTE_SUBFEATURES_SIZE = 4;
			static const int BATCH_MEM_SUBFEATURES_SIZE = 1;
			static const int BATCH_GAUSS_SIZE = 1;

			static const int IMG_WIDTH = 64;
			static const int IMG_HEIGHT = 64;
			static const int MAX_OFFSET = 4;
			static const int NUM_SM = 1; // number of streaming multiprocessors - seems to be better with NUM_SM = 1;

			typedef class BlockIndexing<NUM_SM,
							BLOCK_X, BLOCK_Y, BLOCK_FEATURES,
							BATCH_PIXELS_SIZE_X, BATCH_PIXELS_SIZE_Y,
							1, 1,
							BATCH_FEATURES_SIZE,
							BATCH_COMPUTE_SUBFEATURES_SIZE,
							BATCH_MEM_SUBFEATURES_SIZE,
							BATCH_GAUSS_SIZE,
							IMG_WIDTH, IMG_HEIGHT,
							MAX_OFFSET> BlockIndexingPipelineT;

			BlockIndexingPipelineT::Launch block_indexing;

			dim3 threadsPerBlock = block_indexing.getThreadsPerBlock(I, F, S, img_width, img_height);

			dim3 numBlocks = block_indexing.getBlocksPerGrid(I, F, S, img_width, img_height);

			std::cout << "started fast_gauss_forward_LDS128_kernel" << std::endl;

			std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
			std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

			clock_t start_t = clock();
			fast_gauss_forward_LDS128_kernel<BLOCK_X,BLOCK_Y,BATCH_PIXELS_SIZE_X, BATCH_FEATURES_SIZE, BATCH_COMPUTE_SUBFEATURES_SIZE, BATCH_MEM_SUBFEATURES_SIZE, IMG_WIDTH, IMG_HEIGHT, MAX_OFFSET><<<numBlocks,threadsPerBlock>>>(filtered_images_tex_layered_gpu, filtered_images, filter_offsets_x, filter_offsets_y, filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
			CUDA_POST_KERNEL_CHECK;
			std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
			cudaDeviceSynchronize();

			clock_t end_t = clock();
			std::cout << "fast_gauss_forward_LDS128_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

		}


		if (1) {
			/*static const int BATCH_PIXELS_SIZE_X = 4;
			static const int BATCH_PIXELS_SIZE_Y = 2;

			static const int PIXELS_INTERPOLATION_Dx = 1;
			static const int PIXELS_INTERPOLATION_Dy = 1;

			static const int BLOCK_X = 16/BATCH_PIXELS_SIZE_X;
			static const int BLOCK_Y = 16/BATCH_PIXELS_SIZE_Y;
			static const int BLOCK_FEATURES = 8;

7			static const int BATCH_FEATURES_SIZE = 8;
			static const int BATCH_COMPUTE_SUBFEATURES_SIZE = 4;
			static const int BATCH_MEM_SUBFEATURES_SIZE = 4;
			static const int BATCH_GAUSS_SIZE = 2;
			/*/

			static const int BATCH_PIXELS_SIZE_X = 2;
			static const int BATCH_PIXELS_SIZE_Y = 4;

			static const int PIXELS_INTERPOLATION_Dx = 2;
			static const int PIXELS_INTERPOLATION_Dy = 2;

			static const int BLOCK_X = 32/BATCH_PIXELS_SIZE_X;
			static const int BLOCK_Y = 8/BATCH_PIXELS_SIZE_Y;
			static const int BLOCK_FEATURES = 8; //8

			static const int BATCH_FEATURES_SIZE = 4;
			static const int BATCH_COMPUTE_SUBFEATURES_SIZE = 1;
			static const int BATCH_MEM_SUBFEATURES_SIZE = 2;
			static const int BATCH_GAUSS_SIZE = 2;
			//*/
			static const int IMG_WIDTH = 64;
			static const int IMG_HEIGHT = 64;
			static const int MAX_OFFSET = 8;

			static const int NUM_SM = 1; // number of streaming multiprocessors

			typedef class BlockIndexing<NUM_SM,
							BLOCK_X, BLOCK_Y, BLOCK_FEATURES,
							BATCH_PIXELS_SIZE_X, BATCH_PIXELS_SIZE_Y,
							PIXELS_INTERPOLATION_Dx, PIXELS_INTERPOLATION_Dy,
							BATCH_FEATURES_SIZE,
							BATCH_COMPUTE_SUBFEATURES_SIZE,
							BATCH_MEM_SUBFEATURES_SIZE,
							BATCH_GAUSS_SIZE,
							IMG_WIDTH, IMG_HEIGHT,
							MAX_OFFSET> BlockIndexingPipelineT;

			BlockIndexingPipelineT::Launch block_indexing;

			dim3 threadsPerBlock = block_indexing.getThreadsPerBlock(I, F, S, img_width, img_height);

			dim3 numBlocks = block_indexing.getBlocksPerGrid(I, F, S, img_width, img_height);

			float* filtered_images_with_border;
			{
				std::cout << "started create_input_with_border" << std::endl;

				const int NUM_STREAMS = S;

				cudaStream_t* streams = new cudaStream_t[NUM_STREAMS];

				for (int i = 0; i < NUM_STREAMS; ++i) {
					cudaStreamCreate(&streams[i]);
				}

				clock_t start_t = clock();
				filtered_images_with_border = create_input_with_border(filtered_images, img_width, img_height, I*S, MAX_OFFSET, streams, NUM_STREAMS);
				CUDA_POST_KERNEL_CHECK;
				std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
				cudaDeviceSynchronize();

				clock_t end_t = clock();
				std::cout << "create_input_with_border in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

				for (int i = 0; i < NUM_STREAMS; ++i) {
					cudaStreamDestroy(streams[i]);
				}


				/*
				float* filtered_images_cpu = new float[(img_width + 2*MAX_OFFSET)*( img_height + 2*MAX_OFFSET)* I*S];

				for (int i = 0; i < (img_width + 2*MAX_OFFSET)*( img_height + 2*MAX_OFFSET)* I*S; ++i)
					filtered_images_cpu[i] = -1;

				cudaMemcpy(filtered_images_cpu, filtered_images_with_border, sizeof(int)* (img_width + 2*MAX_OFFSET)*( img_height + 2*MAX_OFFSET)* I*S, cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();

				//for (int i = 0; i < (img_width + 2*MAX_OFFSET)*( img_height + 2*MAX_OFFSET)* I*S; ++i) {
				int index =0;
				for (int i = 0; i < I; ++i) {
					for (int s = 0; s < S; ++s) {
						for (int l =0; l < img_height + 2*MAX_OFFSET; ++l){
							for (int n = 0; n < img_width + 2*MAX_OFFSET; ++n) {
								std::cout << filtered_images_cpu[index++] << " ";
							}
							std::cout << std::endl;
						}
						std::cout << std:: endl << "end of s" << std::endl;
					}
					std::cout << std::endl;

				}
				std::cout << std::endl;
				return;*/
			}

			float* prepared_filter_weights;
			int* prepared_filter_offsets;

			{
				/*float* filter_weights_permut_cpu = new float[S*F*G];

				for (int i = 0; i < S*G*F; ++i) {
					filter_weights_permut_cpu[i] = i;
				}

				cudaMemcpy((float*)filter_weights, filter_weights_permut_cpu, sizeof(float)* S*F*G, cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				for (int i = 0; i < S*G*F; ++i) {
					filter_weights_permut_cpu[i] = -1;
				}*/


				// allocate additional block of data so that double buffering can load valid data on last batch
				static const int OFFSET_BLOCK_MEM_SIZE = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * BATCH_FEATURES_SIZE *  BLOCK_FEATURES;
				static const int WEIGHT_BLOCK_MEM_SIZE = BATCH_COMPUTE_SUBFEATURES_SIZE * BATCH_MEM_SUBFEATURES_SIZE * BATCH_GAUSS_SIZE * PIXELS_INTERPOLATION_Dx*PIXELS_INTERPOLATION_Dy * BATCH_FEATURES_SIZE * BLOCK_FEATURES;

				cudaMalloc(&prepared_filter_weights, sizeof(float) * ( 4*S*G*F + WEIGHT_BLOCK_MEM_SIZE));
				cudaMalloc(&prepared_filter_offsets, sizeof(float) * ( S*G*F + OFFSET_BLOCK_MEM_SIZE));

				dim3 threadsPerBlock(16, 1, 16);
				dim3 numBlocks((int)ceil((F/4)/threadsPerBlock.x),
								(int)ceil(G/threadsPerBlock.y),
								(int)ceil(S/threadsPerBlock.z));

				std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
				std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;

				std::cout << "started copy_permute_weights" << std::endl;

				clock_t start_t = clock();
				perpare_weights_and_offsets<BlockIndexingPipelineT><<<numBlocks,threadsPerBlock>>>(filter_weights, filter_offsets_x, filter_offsets_y, prepared_filter_weights, prepared_filter_offsets, S, G, F/4);


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

			std::cout << "started fast_gauss_forward_pipeline_kernel" << std::endl;

			std::cout << "threadsPerBlock " << threadsPerBlock.x << "," << threadsPerBlock.y << "," << threadsPerBlock.z << std::endl;
			std::cout << "numBlocks " << numBlocks.x << "," << numBlocks.y << "," << numBlocks.z << std::endl;


			clock_t start_t = clock();
			fast_gauss_forward_pipeline_kernel<BlockIndexingPipelineT><<<numBlocks,threadsPerBlock>>>(filtered_images_tex_layered_gpu, filtered_images_with_border, prepared_filter_offsets, prepared_filter_weights, output, I, S, F, G, img_width, img_height, kernel_width, kernel_height);
			CUDA_POST_KERNEL_CHECK;
			std::cout << "waiting for cudaDeviceSynchronize" << std::endl;
			cudaDeviceSynchronize();

			clock_t end_t = clock();
			std::cout << "fast_gauss_forward_pipeline_kernel in " << (((float)(end_t-start_t))/CLOCKS_PER_SEC) << std::endl;

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


