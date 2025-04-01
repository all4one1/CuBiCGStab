#pragma once
#include "cuda_runtime.h"
#include <vector>
#include "CudaExtras.h"

namespace CuCG
{
	extern __device__ double rs_old, rs_new, alpha, beta, buffer, buffer2, omega;

	struct CudaReductionM
	{
		std::vector<unsigned int> grid_v;
		std::vector<unsigned int> N_v;

		#define def_threads 512
		unsigned int N = 0;
		unsigned int steps = 0, threads = def_threads, smem = sizeof(double) * def_threads;

		double* res_array = nullptr;
		double res = 0;
		double** arr = nullptr;
		double* second = nullptr;

		CudaReductionM(unsigned int N, unsigned int thr = def_threads);
		CudaReductionM();
		~CudaReductionM();

		double reduce(double* v1, double* v2, bool withCopy = true, ExtraAction action = ExtraAction::NONE);
		CuGraph make_graph(double* v1, double* v2, bool withCopy, ExtraAction action);
	};
}


