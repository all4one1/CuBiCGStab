#pragma once
#include "CudaExtras.h"

extern __device__ double rs_old, rs_new, alpha, beta, buffer, buffer2, omega;

__global__ void check();
__global__ void check2(double* f, unsigned int N);
__global__ void hadamard_product(double* res, double* v1, double* v2, unsigned int N);
__global__ void matrix_dot_vector(double* res, SparseMatrixCuda M, double* vm, unsigned int N);
__global__ void vector_add_vector(double* res, double* v1, double* v2, unsigned int N);
__global__ void vector_substract_vector(double* res, double* v1, double* v2, unsigned int N);
__global__ void scalar_dot_vector(double* res, double scalar, double* v, unsigned int N);
__global__ void vector_minus_matrix_dot_vector(double* res, double* v, SparseMatrixCuda M, double* vm, unsigned int N);
__global__ void vector_minus_vector(double* res, double* v1, double* v2, unsigned int N, KernelCoefficient choice);
__global__ void vector_add_2vectors(double* res, double* v, double* vs1, double* vs2, unsigned int N, KernelCoefficient choice);
__global__ void vector_set_to_vector(double* res, double* v, unsigned int N);
__global__ void vector_set_to_value(double* res, double scalar, unsigned int N);
