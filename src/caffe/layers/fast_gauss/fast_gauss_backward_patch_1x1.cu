#include "caffe/layers/fast_gauss/fast_gauss_backward_core.hpp"

namespace  caffe {

void fast_gauss_backward_multi_subfeatures_patch_1x1(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, bool SMALLER_WARP_AND_GROUP_K, int BATCH_IMAGES, bool USE_INTERPOLATION, bool SINGLE_SUBFEATURE, FastGaussBackward<float>::CUDAParams& PARAMS){

    if (SMALLER_WARP_AND_GROUP_K) {
        RUN_KERNEL_R4(FastGaussBackwardCUDA, 2, 2, MAX_OFFSET, 4, 4, 2, 2, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS);
    } else {
        RUN_KERNEL_R4(FastGaussBackwardCUDA, 2, 2, MAX_OFFSET, 3, 1, 2, 2, BATCH_IMAGES, USE_INTERPOLATION, SINGLE_SUBFEATURE, PARAMS);
    }
}

}