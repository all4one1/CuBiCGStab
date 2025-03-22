#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <stdio.h>

#include "enums.h"
#include "CudaExtras.h"
#include "ReductionM.h"

namespace CuCG
{
	using std::cout;
	using std::endl;


	__global__ void check()
	{
		printf("device: \n");
		printf("rs_old = %f, rs_new = %f, buffer = %f \n", rs_old, rs_new, buffer);
		printf("alpha = %f, beta = %f, omega = %f \n", alpha, beta, omega);
	}
	__global__ void check2(double* f, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			printf("%i %f \n", i, f[i]);
		}
	}
	

	__global__ void hadamard_product(double* res, double* v1, double* v2, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = v1[i] * v2[i];
		}
	}
	__global__ void matrix_dot_vector(double* res, SparseMatrixCuda M, double* vm, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double s = 0;
			for (int j = M.row[i]; j < M.row[i + 1]; j++)
			{
				s += M.val[j] * vm[M.col[j]];
			}
			res[i] = s;
		}
	}
	__global__ void vector_add_vector(double* res, double* v1, double* v2, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = v1[i] + v2[i];
		}
	}
	__global__ void vector_substract_vector(double* res, double* v1, double* v2, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = v1[i] - v2[i];
		}
	}
	__global__ void scalar_dot_vector(double* res, double scalar, double* v, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = scalar * v[i];
		}
	}
	__global__ void vector_minus_matrix_dot_vector(double* res, double* v, SparseMatrixCuda M, double* vm, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double s = 0;
			for (int j = M.row[i]; j < M.row[i + 1]; j++)
			{
				s += M.val[j] * vm[M.col[j]];
			}
			res[i] = v[i] - s;
		}
	}
	__global__ void vector_minus_vector(double* res, double* v1, double* v2, unsigned int N, KernelCoefficient choice)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double coef;
			switch (choice)
			{
			case CuCG::KernelCoefficient::alpha:
				coef = alpha;
				break;
			case CuCG::KernelCoefficient::omega:
				coef = omega;
				break;
			default:
				break;
			}
			res[i] = v1[i] - coef * v2[i];
		}
	}
	__global__ void vector_add_2vectors(double* res, double* v, double* vs1, double* vs2, unsigned int N, KernelCoefficient choice)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double coef1, coef2;
			switch (choice)
			{
			case KernelCoefficient::beta_and_omega:
				coef1 = beta;
				coef2 = -beta * omega;
				break;
			case KernelCoefficient::alpha_and_omega:
				coef1 = alpha;
				coef2 = omega;
				break;
			default:
				break;
			}
			res[i] = v[i] + coef1 * vs1[i] + coef2 * vs2[i];
		}
	}
	__global__ void vector_set_to_vector(double* res, double* v, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = v[i];
		}
	}
	__global__ void vector_set_to_value(double* res, double scalar, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = scalar;
		}
	}


	struct BiCGSTAB
	{
		double* r = nullptr, * r_hat = nullptr, * p = nullptr, * t = nullptr, * s = nullptr, * v = nullptr;

		double eps = 1e-8;
		double rs_host = 1;
		unsigned int N = 0, Nbytes = 0, k = 0;
		unsigned int threads = 1, blocks = 1;
		CuGraph graph;
		CuCG::CudaReductionM* CR;

		#define KERNEL(func) func<<< blocks, threads>>>

		BiCGSTAB() {};
		BiCGSTAB(unsigned int N_, double* x, double* x0, double* b,
			SparseMatrixCuda& A, CudaLaunchSetup kernel_setting, unsigned int reduction_threads = 256)
		{
			N = N_;
			Nbytes = N * sizeof(double);
			threads = kernel_setting.Block1D.x;
			blocks = kernel_setting.Grid1D.x;

			#define alloc_(ptr) cudaMalloc((void**)&##ptr, Nbytes);  cudaMemset(ptr, 0, Nbytes); 
			alloc_(r); alloc_(r_hat); alloc_(p); alloc_(t); alloc_(s); alloc_(v);

			CR = new CuCG::CudaReductionM(N, reduction_threads);
			make_graph(x, x0, b, A);
		}
		void solve_directly(double* x, double* x0, double* b, SparseMatrixCuda& A)
		{
			double rs_host = 1;  k = 0;
			cudaMemset(x, 0, Nbytes);

			// r = b - Ax
			KERNEL(CuCG::vector_minus_matrix_dot_vector)(r, b, A, x, N);
			// r_hat = r
			KERNEL(CuCG::vector_set_to_vector)(r_hat, r, N);
			// p = r
			KERNEL(CuCG::vector_set_to_vector)(p, r, N);

			// rs = r_hat * r
			CR->reduce(r_hat, r, true, CuCG::ExtraAction::compute_rs_old);

			auto single_iteration = [&]()
			{

				// rs_new = r_hat * r; 		// beta =  (rs_new / rs_old) * (alpha / omega)		// rs_old = rs_new
				CR->reduce(r_hat, r, false, CuCG::ExtraAction::compute_rs_new_and_beta);

				// p = r + beta * ( p - omega * v)
				KERNEL(CuCG::vector_add_2vectors)(p, r, p, v, N, CuCG::KernelCoefficient::beta_and_omega);

				// v = Ap
				KERNEL(CuCG::matrix_dot_vector)(v, A, p, N);

				// alpha = rs_new / (r_hat * v)
				CR->reduce(r_hat, v, false, CuCG::ExtraAction::compute_alpha);

				// s = r - alpha * v
				KERNEL(CuCG::vector_minus_vector)(s, r, v, N, CuCG::KernelCoefficient::alpha);

				// t = A * s
				KERNEL(CuCG::matrix_dot_vector)(t, A, s, N);

				// omega = (t * s) / (t * t)

				CR->reduce(t, s, false, CuCG::ExtraAction::compute_buffer);
				CR->reduce(t, t, false, CuCG::ExtraAction::compute_omega);

				// x = x + alpha * p + omega * s
				KERNEL(CuCG::vector_add_2vectors)(x, x, p, s, N, CuCG::KernelCoefficient::alpha_and_omega);

				// r = s - omega * t
				KERNEL(CuCG::vector_minus_vector)(r, s, t, N, CuCG::KernelCoefficient::omega);
			};


			while (true)
			{
				k++;	if (k > 1000000) break;

				single_iteration();

				// check exit by r^2
				if (k < 20 || k % 50 == 0)
				{
					rs_host = CR->reduce(r, r, true, CuCG::ExtraAction::NONE);
					//if (k > 100000) break;
					if (abs(rs_host) < eps) break;
				}

				//if (k == 20000) break;
				if (k % 1000 == 0) cout << k << " " << abs(rs_host) << endl;
			}

			cout << k << " " << abs(rs_host) << endl;
		}
		void make_graph(double* x, double* x0, double* b, SparseMatrixCuda& A)
		{
			CuCG::KernelCoefficient action;

			// 1. rs_new = r_hat * r; 		// beta =  (rs_new / rs_old) * (alpha / omega)		// rs_old = rs_new
			graph.add_graph_as_node(CR->make_graph(r_hat, r, false, CuCG::ExtraAction::compute_rs_new_and_beta));

			// 2. p = r + beta * ( p - omega * v)
			{
				action = CuCG::KernelCoefficient::beta_and_omega;
				void* args[] = { &p, &r, &p, &v, &N, &action };
				graph.add_kernel_node(threads, blocks, CuCG::vector_add_2vectors, args);
			}

			// 3. v = Ap
			{
				void* args[] = { &v, &A, &p, &N };
				graph.add_kernel_node(threads, blocks, CuCG::matrix_dot_vector, args);
			}

			// 4. alpha = rs_new / (r_hat * v)
			graph.add_graph_as_node(CR->make_graph(r_hat, v, false, CuCG::ExtraAction::compute_alpha));

			// 5. s = r - alpha * v
			{
				action = CuCG::KernelCoefficient::alpha;
				void* args[] = { &s, &r, &v, &N, &action };
				graph.add_kernel_node(threads, blocks, CuCG::vector_minus_vector, args);
			}

			// 6. t = A * s
			{
				void* args[] = { &t, &A, &s, &N };
				graph.add_kernel_node(threads, blocks, CuCG::matrix_dot_vector, args);
			}

			// 7. omega = (t * s) / (t * t)
			graph.add_graph_as_node(CR->make_graph(t, s, false, CuCG::ExtraAction::compute_buffer));
			graph.add_graph_as_node(CR->make_graph(t, t, false, CuCG::ExtraAction::compute_omega));

			// 8. x = x + alpha * p + omega * s
			{
				action = CuCG::KernelCoefficient::alpha_and_omega;
				void* args[] = { &x, &x, &p, &s, &N, &action };
				graph.add_kernel_node(threads, blocks, CuCG::vector_add_2vectors, args);
			}

			// 9. r = s - omega * t
			{
				action = CuCG::KernelCoefficient::omega;
				void* args[] = { &r, &s, &t, &N, &action };
				graph.add_kernel_node(threads, blocks, CuCG::vector_minus_vector, args);
			}

			graph.instantiate();
		}
		void solve_with_graph(double* x, double* x0, double* b, SparseMatrixCuda& A)
		{
			double rs_host = 1;  k = 0;
			cudaMemset(x, 0, Nbytes);

			// r = b - Ax
			KERNEL(CuCG::vector_minus_matrix_dot_vector)(r, b, A, x, N);
			// r_hat = r
			KERNEL(CuCG::vector_set_to_vector)(r_hat, r, N);
			// p = r
			KERNEL(CuCG::vector_set_to_vector)(p, r, N);
			// rs = r_hat * r
			CR->reduce(r_hat, r, false, CuCG::ExtraAction::compute_rs_old);

			while (true)
			{
				k++;	if (k > 1000000) break;
				graph.launch();

				// check exit by r^2
				if (k < 20 || k % 50 == 0)
				{
					rs_host = CR->reduce(r, r, true, CuCG::ExtraAction::NONE);
					//if (k > 100000) break;
					if (abs(rs_host) < eps) break;
				}

				//if (k == 20000) break;
				if (k % 1000 == 0) cout << k << " " << abs(rs_host) << endl;
			}

			cout << k << " " << abs(rs_host) << endl;
		}
	};
}

