#include "CuCG.h"
#include <iostream>
using namespace std;

int main()
{
	// example: 

	int n = 6;	// rank of a square matrix 
	int nval = 24;	// number of non-zero elements of a matrix

	// Example of a sparse matrix
	double sparse_matrix_elements[24] = { 30, 3, 4, 4, 22, 1, 3, 5, 7, 33, 6, 7, 1, 2, 42, 3, 3, 2, 11, 52, 2, 3, 9, 26 }; 
	int column[24] = { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5 };
	int row[7] = { 0, 3, 7, 12, 17, 21, 24 };

	SparseMatrixCuda SMC(n, nval, sparse_matrix_elements, column, row);

	double fh[6] = { 0, 0, 0, 0, 0, 0 };
	double bh[6] = { 1, 2, 3, 3, 2, 1 };
	double* d, * d0, * b;
	cudaMalloc((void**)&d, sizeof(double) * n);
	cudaMalloc((void**)&d0, sizeof(double) * n);
	cudaMalloc((void**)&b, sizeof(double) * n);

	cudaMemcpy(d0, fh, sizeof(double) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(b, bh, sizeof(double) * n, cudaMemcpyHostToDevice);
	CudaLaunchSetup kernel(6);

	// solver usage:
	CuCG::BiCGSTAB solver_cg(6, d, d0, b, SMC, kernel);
	solver_cg.solve_with_graph(d, d0, b, SMC); // #1 
	//solver_cg.solve_directly(d, d0, b, SMC); // #2 


	// check:
	cudaMemcpy(fh, d, sizeof(double) * n, cudaMemcpyDeviceToHost);
	cout << "cuda test:   ";	for (int i = 0; i < n; i++)		cout << fh[i] << " ";	 cout << endl;
	double cg[6] =	{ 0.1826929218e-1,	0.7636750835e-1,	0.5570467736e-1,	0.6371099009e-1,	0.2193724104e-1,	0.2351661001e-1 };
	cout << "x should be: ";	for (int i = 0; i < n; i++)		cout << cg[i] << " ";	cout << endl;

	cudaFree(d);	cudaFree(d0);	cudaFree(b);
    return 0;
}
