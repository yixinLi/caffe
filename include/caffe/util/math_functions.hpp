// Copyright 2013 Yangqing Jia

#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <mkl.h>
#include <cublas_v2.h>

namespace caffe {

// Decaf gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y, const int incx = 1, const int incy = 1);

template <typename Dtype>
void caffe_gpu_copy(const int N, const Dtype *X, Dtype *Y, const int incx = 1, const int incy = 1);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void caffe_vRngUniform(const int n, Dtype* r, const Dtype a, const Dtype b);

template <typename Dtype>
void caffe_vRngGaussian(const int n, Dtype* r, const Dtype a,
    const Dtype sigma);

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void caffe_cpu_tanh(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_tanh(const int n, const Dtype* x, Dtype* y);
    
template <typename Dtype>
void caffe_cpu_sigm(const int n, const Dtype* x, Dtype* y);
    
template <typename Dtype>
void caffe_gpu_sigm(const int n, const Dtype* x, Dtype* y);
    
template <typename Dtype>
void caffe_cpu_sech2(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_sech2(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

template <typename Dtype>
Dtype caffe_gpu_asum(const int n, const Dtype* x);

template <typename Dtype>
void caffe_cpu_diagmat(const int n, const Dtype beta, Dtype* x);
    
template <typename Dtype>
void caffe_diagaxpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);
    
//template <typename Dtype>
//void caffe_gpu_diagaxpy(const int N, const float alpha, const float* X, float* Y);
//    
//template <typename Dtype>
//void caffe_gpu_diagaxpy(const int N, const double alpha, const double* X, double* Y);

template <typename Dtype>
void caffe_gpu_matrix_diag_add_constant(const int N, Dtype* a,                                                   const Dtype lambdas);

template <typename Dtype>
void caffe_abs(const int n, const Dtype* a, Dtype* y);


}  // namespace caffe


#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
