#ifndef CAFFE_UTIL_FAST_GAUSS_FORWARD_H_
#define CAFFE_UTIL_FAST_GAUSS_FORWARD_H_


namespace caffe {
#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void fast_gauss_forward(const Dtype* filtered_images, const int* filter_offsets_x, const int* filter_offsets_y, const float* filter_offsets_float_x, const float* filter_offsets_float_y,
						const Dtype* filter_weights, Dtype* output,
						const int I, const int S, const int F, const int G,
						const int img_width, const int img_height,
						const int kernel_width, const int kernel_height, cudaStream_t streamId = 0);
#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_FAST_GAUSS_FORWARD_H_
