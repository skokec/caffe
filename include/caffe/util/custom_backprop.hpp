#ifndef CAFFE_UTIL_CUSTOM_BACKPROP_H_
#define CAFFE_UTIL_CUSTOM_BACKPROP_H_


namespace caffe {
#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void filterActs_YxX_color(const Dtype* images, const Dtype* error, const Dtype* filters, Dtype* output,
										const int I, const int S, const int F, const int G,
										const int subfeat_i, const int feat_i, const int gauss_i,
										const int img_width, const int img_height,
										const int error_width, const int error_height,
										const int kernel_width, const int kernel_height,
										const int padding, const int stride, cudaStream_t streamId = 0);
#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_CUSTOM_BACKPROP_H_
