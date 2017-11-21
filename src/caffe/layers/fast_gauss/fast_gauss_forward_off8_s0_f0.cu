#include "caffe/layers/fast_gauss/fast_gauss_forward_core.hpp"

namespace  caffe {

void fast_gauss_forward_float_off_8_single_feat_0_single_subfeat_0(int IMG_PATCH_SIZE_W, int IMG_PATCH_SIZE_H, int MAX_OFFSET, int WARP_PIXELS_X, int WARP_PIXELS_Y, int BLOCK_IMAGES, int USE_INTERPOLATION, FastGaussForward<float>::CUDAParams& PARAMS){

    RUN_KERNEL_R4(FastGaussForwardCUDA, IMG_PATCH_SIZE_W, IMG_PATCH_SIZE_H, 8, WARP_PIXELS_X, WARP_PIXELS_Y, BLOCK_IMAGES, USE_INTERPOLATION, false, false, PARAMS);
}

}