#include <stdio.h>
#include "matrix_mul.h"
#include <ctime>

// Thread block size
#define BLOCK_SIZE 16

#define MULTI_HILO
// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, int, float*);
__global__ void Muld(float*, float*, int, int, float*);

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B


void Mul___(float* A, float* B, int hA, int wA, int wB, float* C)
{
	int size;

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	cudaMalloc((void**)&Ad, size);
	std::clock_t start = std::clock();
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	std::clock_t end = std::clock();
	double Ttx1 = ((double)(end - start)) / CLOCKS_PER_SEC;

	float* Bd;
	size = wA * wB * sizeof(float);
	cudaMalloc((void**)&Bd, size);
	start = std::clock();
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
	end = std::clock();
	double Ttx2 = ((double)(end - start)) / CLOCKS_PER_SEC;

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

#ifndef MULTI_HILO
	Muld<<<1, 1>>>(Ad, Bd, hA, wA, wB, Cd);
	start = std::clock();
	Muld<<<1, 1>>>(Ad, Bd, hA, wA, wB, Cd);
	cudaDeviceSynchronize();
	end = std::clock();
	double Tkrnl = ((double)(end - start)) / CLOCKS_PER_SEC;
#endif

	// Compute the execution configuration assuming
	// the matrix dimensions are multiples of BLOCK_SIZE
	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	if(wB % dimBlock.x != 0) xSum = 1;
	if(hA % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((wB / dimBlock.x) + xSum, (hA / dimBlock.y) + ySum);

	// Launch the device computation
#ifdef MULTI_HILO
	Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);
	start = std::clock();
	Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);
	cudaDeviceSynchronize();
	end = std::clock();
	double Tkrnl = ((double)(end - start)) / CLOCKS_PER_SEC;
#endif

	// Read C from the device
	start = std::clock();
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
	end = std::clock();
	double Ttx3 = ((double)(end - start)) / CLOCKS_PER_SEC;

	double BWtx1 = (hA * wA * sizeof(float) / 10e6) / Ttx1;
	double BWtx2 = (wA * wB * sizeof(float) / 10e6) / Ttx2;
	double Perfkrnl = (2 * hA * wA * wB / 10e6) / Tkrnl;
	double BWtx3 = (hA * wB * sizeof(float) / 10e6) / Ttx3;

	//printf("%s; %s; %s; %s; %s; %s; %s; %s;", "s", "s", "s", "s", "GB/s", "GB/s", "GFLOPS/s", "GB/s");
	//printf("\n");
	printf("%f; %f; %f; %f; %f; %f; %f; %f;", Ttx1, Ttx2, Tkrnl, Ttx3, BWtx1, BWtx2, Perfkrnl, BWtx3);
	printf("\n");

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}

__global__ void Muld(float* A, float* B, int hA, int wA, int wB, float* C)
{
    for (int i=0; i < hA; i++)
		for (int j=0; j < wB; j++){
			C[i*wB+j] = 0.0;
			for (int k=0; k<wA; k++){
				C[i*wB+j] += A[i*wA+k]*B[k*wB+j];
			}
		}
}

__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    
	float total = 0.0f;
	if(row < wA && col < wB) {
		for(int i = 0; i < wA; ++i) {
			total += A[i + row * wA] * B[i * wB + col];
		}
		C[row * wB + col] = total;
	}
}

#if 0
// Device multiplication function called by Mul()
// Compute C = A * B
// wA is the width of A
// wB is the width of B
__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = ...;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = ...;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// The element of the block sub-matrix that is computed
	// by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B required to
	// compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Shared memory for the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Shared memory for the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from global memory to shared memory;
		// each thread loads one element of each matrix
		As[ty][tx] = A[...];
		Bs[ty][tx] = B[...];
		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			....

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to global memory;
	// each thread writes one element
	...
}
#endif
