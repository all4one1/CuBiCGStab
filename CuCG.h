#include "CudaExtras.h"
#include "ReductionM.h"

namespace CuCG
{
	struct BiCGSTAB
	{
		double* r = nullptr, * r_hat = nullptr, * p = nullptr, * t = nullptr, * s = nullptr, * v = nullptr;

		double eps = 1e-8;
		double rs_host = 1;
		unsigned int N = 0, Nbytes = 0, k = 0;
		unsigned int threads = 1, blocks = 1;
		CuGraph graph;
		CudaReductionM* CR;

		BiCGSTAB::BiCGSTAB();
		BiCGSTAB::BiCGSTAB(unsigned int N_, double* x, double* x0, double* b,
			SparseMatrixCuda& A, CudaLaunchSetup kernel_setting, unsigned int reduction_threads = 256);
		void BiCGSTAB::solve_directly(double* x, double* x0, double* b, SparseMatrixCuda& A);
		void BiCGSTAB::make_graph(double* x, double* x0, double* b, SparseMatrixCuda& A);
		void BiCGSTAB::solve_with_graph(double* x, double* x0, double* b, SparseMatrixCuda& A);
	};
}

