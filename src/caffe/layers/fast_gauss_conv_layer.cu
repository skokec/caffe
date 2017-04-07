#ifdef USE_CUDNN
#include <vector>
#include <memory>

#include "caffe/layers/gauss_conv_layer.hpp"

#include "caffe/util/math_functions_extra.hpp"
#include "caffe/util/custom_cub.cuh"

namespace caffe {

__global__ void sync_fast_gauss_conv_groups() { }

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	caffe::fast_gauss_forward<Dtype>(NULL, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0);

}

template <typename Dtype>
void FastAproxGaussianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(FastAproxGaussianConvLayer);


}  // namespace caffe
#endif
