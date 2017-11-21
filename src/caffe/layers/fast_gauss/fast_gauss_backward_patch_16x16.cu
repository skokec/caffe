#include "caffe/layers/fast_gauss/fast_gauss_backward_core.hpp"

namespace  caffe {

void fast_gauss_backward_multi_subfeatures_patch_16x16(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackward<float>::CUDAParams& PARAMS){

    RUN_KERNEL_R5(FastGaussBackwardCUDA, 16, 16, MAX_OFFSET, SMALLER_WARP_AND_GROUP_K, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS);
}

}