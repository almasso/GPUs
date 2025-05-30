#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

static struct timeval tv0;
double getMicroSeconds()
{
	double t;
	gettimeofday(&tv0, (struct timezone*)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);

	return (t);
}

void init_seed()
{
	int seedi=1;
	FILE *fd;

	/* Generated random values between 0.00 - 1.00 */
	fd = fopen("/dev/urandom", "r");
	fread( &seedi, sizeof(int), 1, fd);
	fclose (fd);
	srand( seedi );
}

void init1Drand(float *buffer, int n)
{
	int i, j;

	for (i=0; i<n; i++)
		for(j=0; j<n; j++)
			buffer[i*n+j] = 500.0*(float(rand())/RAND_MAX)-500.0; /* [-500 500]*/
}

float *getmemory1D( int nx )
{
	int i;
	float *buffer;

	if( (buffer=(float *)malloc(nx*sizeof(float *)))== NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}

	for( i=0; i<nx; i++ )
		buffer[i] = 0.0;

	return( buffer );
}



/********************************************************************************/
/********************************************************************************/

/*
 * Traspose 1D version
 */
void transpose1D(float *in, float *out, int n)
{
	int i, j;

	for(j=0; j < n; j++) 
		for(i=0; i < n; i++) 
			out[j*n+i] = in[i*n+j]; 
}

/*
 * Traspose CUDA version
 */

#define NTHREADS1D 32

__global__ void transpose_device(float *in, float *out, int rows, int cols) 
{ 
	int i, j, iT, jT;  
	i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float tmpMatrix[NTHREADS1D + 1][NTHREADS1D];   

	if (i<cols && j < rows)
        tmpMatrix[threadIdx.y][threadIdx.x] = in[j * cols + i];

	iT = blockIdx.y * blockDim.y + threadIdx.x;
	jT = blockIdx.x * blockDim.x + threadIdx.y;
    __syncthreads();
    if (i<cols && j < rows)
        out[jT * cols + iT] = tmpMatrix[threadIdx.x][threadIdx.y];

}

int check(float *GPU, float *CPU, int n)
{
	int i;

	for (i=0; i<n; i++)
		if(GPU[i]!=CPU[i]){
			printf("(%i) %f %f\n", i, GPU[i], CPU[i]);
			return(1);
	}
	return(0);
}

int main(int argc, char **argv)
{
	int n;
	float *array1D_trans;
	float *darray1D, *darray1D_trans;

	double t0, t1;

	if (argc==2)
		n = atoi(argv[1]);
	else {
		n = 8192;
		printf("./exec n (by default n=%i)\n", n);
	}
	
	/* Initizalization */
	init_seed();

	/* Transpose 1D version */
	cudaMallocManaged((void**)&darray1D, n*n*sizeof(float));
	array1D_trans = (float*)malloc(sizeof(float)*n*n);
	init1Drand(darray1D, n);
	t0 = getMicroSeconds();
	transpose1D(darray1D, array1D_trans, n);
	printf("Transpose version 1D: %f MB/s\n", n*n*sizeof(float)/((getMicroSeconds()-t0)/1000000)/1024/1024);


	/* CUDA vesion */
	cudaMallocManaged((void**)&darray1D_trans, n*n*sizeof(float));

	dim3 dimBlock(NTHREADS1D, NTHREADS1D);
    int eX = 0, eY = 0;
    if(n % dimBlock.x != 0) eX = 1;
    if(n % dimBlock.y != 0) eY = 1;
	dim3 dimGrid(n/dimBlock.x + eX, n/dimBlock.y + eY);

	t0 = getMicroSeconds();
	transpose_device<<<dimGrid,dimBlock>>>(darray1D, darray1D_trans, n, n);	
	cudaThreadSynchronize();
	t1 = getMicroSeconds();

	
	printf("Transpose kernel version: %f MB/s tKernel=%f (us)\n", n*n*sizeof(float)/((getMicroSeconds()-t0)/1000000)/1024/1024, (t1-t0)/1000000);
	

	if (check(darray1D_trans, array1D_trans, n*n))
		printf("Transpose CPU-GPU differs!!\n");

	cudaFree(darray1D);
	cudaFree(darray1D_trans);

	return(0);
}
