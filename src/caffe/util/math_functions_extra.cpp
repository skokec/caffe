#include "caffe/util/math_functions_extra.hpp"

namespace caffe {

template <>
void caffe_cpu_copy_strided<float>(const int N, const float* X, int incx, float*Y, int incy) {
	cblas_scopy(N, X, incx, Y, incy);
}

template <>
void caffe_cpu_copy_strided<double>(const int N, const double* X, int incx, double *Y, int incy) {
	cblas_dcopy(N, X, incx, Y, incy);
}

template <typename Dtype>
void caffe_cpu_sum(const int n, const Dtype* x, Dtype* y, int m) {
	if (m <= 0)
		m = n;
	int num_segments = n/m;

	int input_offset = 0;
	for (int i = 0; i < num_segments; ++i) {

		Dtype sum_value = 0;

		for (int j = 0; j < m; ++j) {
			sum_value += x[input_offset];
			input_offset++;
		}

		y[i] += sum_value;
	}
}

template void caffe_cpu_sum(const int n, const double* x, double* y, int m);
template void caffe_cpu_sum(const int n, const float* x, float* y, int m);

}  // namespace caffe
