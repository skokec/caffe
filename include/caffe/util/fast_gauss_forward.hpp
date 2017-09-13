#ifndef CAFFE_UTIL_FAST_GAUSS_FORWARD_H_
#define CAFFE_UTIL_FAST_GAUSS_FORWARD_H_


namespace caffe {
#ifndef CPU_ONLY  // GPU

enum { FAST_GAUSS_PARAM_SGF, FAST_GAUSS_PARAM_FGS};

template <typename Dtype>
void fast_gauss_forward(const Dtype* filtered_images,
                        const Dtype* filter_offsets_float_x, const Dtype* filter_offsets_float_y, const Dtype* filter_weights, int PARAM_FORMAT,
                        Dtype* output,
						const int I, const int S, const int F, const int G,
						const int img_width, const int img_height,
						const int kernel_width, const int kernel_height, const bool use_interpolation,
						Dtype* prepared_filtered_images, size_t* prepared_filtered_images_size,
						Dtype* prepared_filter_weights, size_t* prepared_filter_weights_size,
						int* prepared_filter_offsets, size_t* prepared_filter_offsets_size,
						Dtype* prepared_filter_offsets_and_weights, cudaStream_t streamId = 0);
#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_FAST_GAUSS_FORWARD_H_
