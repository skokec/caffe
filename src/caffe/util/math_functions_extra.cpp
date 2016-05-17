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

}  // namespace caffe
