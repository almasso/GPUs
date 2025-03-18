#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <algorithm>
#include "png_io.h"
#include "routinesGPU.h"

#define BLOCK_SIZE 32

__global__ void noiseReduction(uint8_t* im, float* NR, int width, int height) {
	int tx = threadIdx.x, ty = threadIdx.y;
	int i = ty + blockIdx.y * blockDim.y;
	int j = tx + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	__shared__ float sNR[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

	sNR[ty + 2][tx + 2] = im[i * width + j];

	if(ty < 2) {
		sNR[ty][tx + 2] = im[(i - 2) * width + j];
		sNR[ty + BLOCK_SIZE + 2][tx + 2] = im[(i + BLOCK_SIZE) * width + j];
	}
	if(tx < 2) {
		sNR[ty+ 2][tx] = im[i * width + (j - 2)];
		sNR[ty+ 2][tx + BLOCK_SIZE + 2] = im[i * width + (j + BLOCK_SIZE)];
	}
	__syncthreads();

	// Noise reduction
	float res = (
	  2.0*sNR[ty][tx]     +  4.0*sNR[ty][tx + 1]     +   5.0*sNR[ty][tx + 2]     +  4.0*sNR[ty][tx + 3]     +  2.0*sNR[ty][tx + 4]
	+ 4.0*sNR[ty + 1][tx] +  9.0*sNR[ty + 1][tx + 1] +  12.0*sNR[ty + 1][tx + 2] +  9.0*sNR[ty + 1][tx + 3] +  4.0*sNR[ty + 1][tx + 4]
	+ 5.0*sNR[ty + 2][tx] + 12.0*sNR[ty + 2][tx + 1] +  15.0*sNR[ty + 2][tx + 2] + 12.0*sNR[ty + 2][tx + 3] +  5.0*sNR[ty + 2][tx + 4]
	+ 4.0*sNR[ty + 3][tx] +  9.0*sNR[ty + 3][tx + 1] +  12.0*sNR[ty + 3][tx + 2] +  9.0*sNR[ty + 3][tx + 3] +  4.0*sNR[ty + 3][tx + 4]
	+ 2.0*sNR[ty + 4][tx] +  4.0*sNR[ty + 4][tx + 1] +   5.0*sNR[ty + 4][tx + 2] +  4.0*sNR[ty + 4][tx + 3] +  2.0*sNR[ty + 4][tx + 4])
	/159.0;

	NR[i * width + j] = res;
}

__global__ void gradientX(float* Gx, float* NR, int width, int height) {
	int tx = threadIdx.x, ty = threadIdx.y;
	int i = ty + blockIdx.y * blockDim.y;
	int j = tx + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	__shared__ float sGx[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

	sGx[ty + 2][tx + 2] = NR[i * width + j];

	if(ty < 2) {
		sGx[ty][tx + 2] = NR[(i - 2) * width + j];
		sGx[ty + BLOCK_SIZE + 2][tx + 2] = NR[(i + BLOCK_SIZE) * width + j];
	}
	if(tx < 2) {
		sGx[ty+ 2][tx] = NR[i * width + (j - 2)];
		sGx[ty+ 2][tx + BLOCK_SIZE + 2] = NR[i * width + (j + BLOCK_SIZE)];
	}
	__syncthreads();

	float res = (
		  1.0*sGx[ty][tx]     +  2.0*sGx[ty][tx + 1]     +  (-2.0)*sGx[ty][tx + 3]     + (-1.0)*sGx[ty][tx + 4] 
		+ 4.0*sGx[ty + 1][tx] +  8.0*sGx[ty + 1][tx + 1] +  (-8.0)*sGx[ty + 1][tx + 3] + (-4.0)*sGx[ty + 1][tx + 4]
		+ 6.0*sGx[ty + 2][tx] + 12.0*sGx[ty + 2][tx + 1] + (-12.0)*sGx[ty + 2][tx + 3] + (-6.0)*sGx[ty + 2][tx + 4]
		+ 4.0*sGx[ty + 3][tx] +  8.0*sGx[ty + 3][tx + 1] +  (-8.0)*sGx[ty + 3][tx + 3] + (-4.0)*sGx[ty + 3][tx + 4]
		+ 1.0*sGx[ty + 4][tx] +  2.0*sGx[ty + 4][tx + 1] +  (-2.0)*sGx[ty + 4][tx + 3] + (-1.0)*sGx[ty + 4][tx + 4]);
	
	Gx[i*width+j] = res;
}

__global__ void gradientY(float* Gy, float* NR, int width, int height) {
	int tx = threadIdx.x, ty = threadIdx.y;
	int i = ty + blockIdx.y * blockDim.y;
	int j = tx + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	__shared__ float sGy[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

	sGy[ty + 2][tx + 2] = NR[i * width + j];

	if(ty < 2) {
		sGy[ty][tx + 2] = NR[(i - 2) * width + j];
		sGy[ty + BLOCK_SIZE + 2][tx + 2] = NR[(i + BLOCK_SIZE) * width + j];
	}
	if(tx < 2) {
		sGy[ty+ 2][tx] = NR[i * width + (j - 2)];
		sGy[ty+ 2][tx + BLOCK_SIZE + 2] = NR[i * width + (j + BLOCK_SIZE)];
	}
	__syncthreads();

	float res = (
		  (-1.0)*sGy[ty][tx]     + (-4.0)*sGy[ty][tx + 1]     +  (-6.0)*sGy[ty][tx + 2]     + (-4.0)*sGy[ty][tx + 3]     + (-1.0)*sGy[ty][tx + 4]
		+ (-2.0)*sGy[ty + 1][tx] + (-8.0)*sGy[ty + 1][tx + 1] + (-12.0)*sGy[ty + 1][tx + 2] + (-8.0)*sGy[ty + 1][tx + 3] + (-2.0)*sGy[ty + 1][tx + 4]
		+    2.0*sGy[ty + 3][tx] +   8.0*sGy[ty + 3][tx + 1]  +   12.0*sGy[ty + 3][tx + 2]  +   8.0*sGy[ty + 3][tx + 3]  +   2.0*sGy[ty + 3][tx + 4]
		+    1.0*sGy[ty + 4][tx] +   4.0*sGy[ty + 4][tx + 1]  +    6.0*sGy[ty + 4][tx + 2]  +   4.0*sGy[ty + 4][tx + 3]  +   1.0*sGy[ty + 4][tx + 4]);

	Gy[i*width+j] = res;
}

__global__ void gradient(float* G, float* Gx, float* Gy, int width, int height) {
	int tx = threadIdx.x, ty = threadIdx.y;
	int i = ty + blockIdx.y * blockDim.y;
	int j = tx + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	G[i*width+j] = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
}

__global__ void phiKernel(float* phi, float* Gx, float* Gy, int width, int height) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if((i < 2 || i > height - 3) || (j < 2 || j > width - 3)) return;

	float PI = 3.141593;
	
	phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

	if(fabs(phi[i*width+j])<=PI/8 )
		phi[i*width+j] = 0;
	else if (fabs(phi[i*width+j])<= 3*(PI/8))
		phi[i*width+j] = 45;
	else if (fabs(phi[i*width+j]) <= 5*(PI/8))
		phi[i*width+j] = 90;
	else if (fabs(phi[i*width+j]) <= 7*(PI/8))
		phi[i*width+j] = 135;
	else phi[i*width+j] = 0;
}

__global__ void edgeDetection(uint8_t* pedge, float* G, float* phi, int width, int height) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if((i < 3 || i > height - 4) || (j < 3 || j > width - 4)) return;

	if(phi[i*width+j] == 0) {
		if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
			pedge[i*width+j] = 1;
	} else if(phi[i*width+j] == 45) {
		if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
			pedge[i*width+j] = 1;

	} else if(phi[i*width+j] == 90) {
		if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
			pedge[i*width+j] = 1;

	} else if(phi[i*width+j] == 135) {
		if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
			pedge[i*width+j] = 1;
	}
	
}

__global__ void hysteresis(uint8_t* image_out, uint8_t* pedge, float* G, int width, int height, float level) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	
	if((i < 3 || i > height - 4) || (j < 3 || j > width - 4)) return;
	
	int ii, jj;
	float lowthres = level/2, hithres = 2 *(level);

	if(G[i*width+j]>hithres && pedge[i*width+j]) {
		image_out[i*width+j] = 255;
	}
	else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres) {
		// check neighbours 3x3
		for (ii=-1;ii<=1; ii++)
			for (jj=-1;jj<=1; jj++)
				if (G[(i+ii)*width+j+jj]>hithres) {
					image_out[i*width+j] = 255;
				}
	}
	
}

__global__ void houghKernel(uint8_t* im, uint32_t* accumulators, int width, int height, int accu_width, int accu_height, 
	float* sin_table, float* cos_table) 
{
	int theta;

	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	float center_x = width/2.0; 
	float center_y = height/2.0;

	if( im[ (i*width) + j] > 250 ) { // Pixel is edge
		for(theta=0;theta<180;theta++) {  
			float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
			atomicAdd(&accumulators[(int)((round(rho + hough_h) * 180.0)) + theta], 1);
		} 
	} 
	
}

__global__ void getLinesKernel(int threshold, uint32_t* accumulators, int accu_width, int accu_height, int width, int height, 
	float* sin_table, float* cos_table, int* x1_lines, int* y1_lines, int* x2_lines, int* y2_lines, int* lines) 
{
    int rho = blockIdx.y * blockDim.y + threadIdx.y;
    int theta = blockIdx.x * blockDim.x + threadIdx.x;

    if (rho >= accu_height || theta >= accu_width) return;

    uint32_t accuVote = accumulators[rho * accu_width + theta];
    if (accuVote < threshold) return;

    bool is_max = true;
    for (int ii = -4; ii <= 4 && is_max; ii++) {
        for (int jj = -4; jj <= 4; jj++) {
            int r = rho + ii;
            int t = theta + jj;
            if (r >= 0 && r < accu_height && t >= 0 && t < accu_width) {
                if (accumulators[r * accu_width + t] > accuVote) {
                    is_max = false;
                    break;
                }
            }
        }
    }
    if (!is_max) return;

    int x1, y1, x2, y2;
    if (theta >= 45 && theta <= 135) {
        x1 = (theta > 90) ? width / 2 : 0;
        x2 = (theta > 90) ? width : width * 2 / 5;
        y1 = ((rho - accu_height / 2) - ((x1 - width / 2) * cos_table[theta])) / sin_table[theta] + (height / 2);
        y2 = ((rho - accu_height / 2) - ((x2 - width / 2) * cos_table[theta])) / sin_table[theta] + (height / 2);
    } else {
        y1 = 0;
        y2 = height;
        x1 = ((rho - accu_height / 2) - ((y1 - height / 2) * sin_table[theta])) / cos_table[theta] + (width / 2);
        x2 = ((rho - accu_height / 2) - ((y2 - height / 2) * sin_table[theta])) / cos_table[theta] + (width / 2);
    }

    int index = atomicAdd(lines, 1);
    x1_lines[index] = x1;
    y1_lines[index] = y1;
    x2_lines[index] = x2;
    y2_lines[index] = y2;
}

void canny(uint8_t *im, uint8_t *image_out, int height, int width, float level) 
{
	float* NR, *G, *Gx, *Gy, *phi;
	uint8_t* imTmp, *pedge, *imageoutTmp;

	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	if(width % dimBlock.x != 0) xSum = 1;
	if(height % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((width / dimBlock.x) + xSum, (height / dimBlock.y) + ySum);

	// Pasamos a memoria la imagen y un NR temporal
	cudaMalloc((void**)&NR, width * height * sizeof(float));
	cudaMalloc((void**)&imTmp, width * height * sizeof(uint8_t));
	cudaMemcpy(imTmp, im, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
	noiseReduction<<<dimGrid,dimBlock>>>(imTmp, NR, width, height);
	cudaDeviceSynchronize();
	// En cuanto dejemos de usar la imagen, la liberamos de memoria
	cudaFree(imTmp);

	cudaStream_t streams[2];
	for(int i = 0; i < 2; ++i)
		cudaStreamCreate(&streams[i]);
	
	dim3 multipleStreamsDimGrid(dimGrid.x/2, dimGrid.y/2);
	
	// Reservamos memoria para el resto de variables que vamos a utilizar
	cudaMalloc((void**)&G, width * height * sizeof(float));
	cudaMalloc((void**)&Gx, width * height * sizeof(float));
	cudaMalloc((void**)&Gy, width * height * sizeof(float));
	cudaMalloc((void**)&phi, width * height * sizeof(float));
	gradientX<<<dimGrid, dimBlock, 0, streams[0]>>>(Gx, NR, width, height);
	gradientY<<<dimGrid, dimBlock, 0, streams[1]>>>(Gy, NR, width, height);
	cudaStreamSynchronize(streams[0]);
	cudaStreamSynchronize(streams[1]);
	
	gradient<<<dimGrid, dimBlock, 0, streams[0]>>>(G, Gx, Gy, width, height);
	phiKernel<<<dimGrid, dimBlock, 0, streams[1]>>>(phi, Gx, Gy, width, height);
	cudaStreamSynchronize(streams[0]);
	cudaStreamSynchronize(streams[1]);
	
	// Liberamos las que ya no vayamos a utilizar
	for(int i = 0; i < 2; ++i) 
		cudaStreamDestroy(streams[i]);
	cudaFree(NR);
	cudaFree(Gx);
	cudaFree(Gy);

	// Reservamos el pedge
	cudaMalloc((void**)&pedge, width * height * sizeof(uint8_t));
	cudaMemset(pedge, 0, width * height * sizeof(uint8_t));
	edgeDetection<<<dimGrid,dimBlock>>>(pedge, G, phi, width, height);
	cudaDeviceSynchronize();
	// Y liberamos phi que ya no se va a usar
	cudaFree(phi);

	// Reservamos memoria para la imagen final
	cudaMalloc((void**)&imageoutTmp, width * height * sizeof(uint8_t));
	cudaMemset(imageoutTmp, 0, width * height * sizeof(uint8_t));
	hysteresis<<<dimGrid,dimBlock>>>(imageoutTmp, pedge, G, width, height, level);
	cudaDeviceSynchronize();
	// Y la pasamos a memoria física una vez la hayamos calculado.
	cudaMemcpy(image_out, imageoutTmp, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree(G);
	cudaFree(pedge);
	cudaFree(imageoutTmp);
}

void houghTransform(uint8_t* im, uint32_t* accumulators, int width, int height, int accu_width, int accu_height, 
	float* sin_table, float* cos_table) 
{
	uint8_t* imTmp;
	uint32_t* accuTmp;
	float* sinTTmp, *cosTTmp;

	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	if(width % dimBlock.x != 0) xSum = 1;
	if(height % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((width / dimBlock.x) + xSum, (height / dimBlock.y) + ySum);

	cudaMalloc((void**)&imTmp, width * height * sizeof(uint8_t));
	cudaMalloc((void**)&accuTmp, accu_width * accu_height * sizeof(uint32_t));
	cudaMalloc((void**)&sinTTmp, 180 * sizeof(float));
	cudaMalloc((void**)&cosTTmp, 180 * sizeof(float));
	
	cudaMemcpy(imTmp, im, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(sinTTmp, sin_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cosTTmp, cos_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(accuTmp, 0, accu_width * accu_height * sizeof(uint32_t));

	houghKernel<<<dimGrid, dimBlock>>>(imTmp, accuTmp, width, height, accu_width, accu_height, sinTTmp, cosTTmp);
	cudaDeviceSynchronize();
	cudaMemcpy(accumulators, accuTmp, accu_width * accu_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaFree(sinTTmp);
	cudaFree(cosTTmp);
	cudaFree(imTmp);
	cudaFree(accuTmp);
}

void getLines(int threshold, uint32_t* accumulators, int accu_width, int accu_height, int width, int height, 
	float* sin_table, float* cos_table, int* x1, int* y1, int* x2, int* y2, int* lines) 
{
	uint32_t* accuTmp;
	float* sinTTmp, *cosTTmp;
	int* x1Tmp, *y1Tmp, *x2Tmp, *y2Tmp, *linesTmp;

	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int maximumWidth = std::max(width, accu_width);
	int maximumHeight = std::max(height, accu_height);
	if(maximumWidth % dimBlock.x != 0) xSum = 1;
	if(maximumHeight % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((maximumWidth / dimBlock.x) + xSum, (maximumHeight / dimBlock.y) + ySum);

	cudaMalloc((void**)&accuTmp, accu_width * accu_height * sizeof(uint32_t));
	cudaMalloc((void**)&sinTTmp, 180 * sizeof(float));
	cudaMalloc((void**)&cosTTmp, 180 * sizeof(float));
	cudaMalloc((void**)&x1Tmp, 10 * sizeof(int));
	cudaMalloc((void**)&y1Tmp, 10 * sizeof(int));
	cudaMalloc((void**)&x2Tmp, 10 * sizeof(int));
	cudaMalloc((void**)&y2Tmp, 10 * sizeof(int));
	cudaMalloc((void**)&linesTmp, sizeof(int));

	cudaMemcpy(accuTmp, accumulators, accu_width * accu_height * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(sinTTmp, sin_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cosTTmp, cos_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(linesTmp, lines, sizeof(int), cudaMemcpyHostToDevice);
	
	getLinesKernel<<<dimGrid, dimBlock>>>(threshold, accuTmp, accu_width, accu_height, width, height, sinTTmp, cosTTmp, x1Tmp, y1Tmp, x2Tmp, y2Tmp, linesTmp);
	cudaDeviceSynchronize();

	cudaMemcpy(x1, x1Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y1, y1Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(x2, x2Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y2, y2Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(lines, linesTmp, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(accuTmp);
	cudaFree(sinTTmp);
	cudaFree(cosTTmp);
	cudaFree(x1Tmp);
	cudaFree(y1Tmp);
	cudaFree(x2Tmp);
	cudaFree(y2Tmp);
	cudaFree(linesTmp);
}

void lane_assist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	float level = 1000.0f;
	int threshold = width > height ? width/6 : height / 6;

	canny(im, imEdge, height, width, level);
	write_png_fileBW("out_edges.png", imEdge, width, height);

	houghTransform(imEdge, accum, width, height, accu_width, accu_height, sin_table, cos_table);

	getLines(threshold, accum, accu_width, accu_height, width, height, sin_table, cos_table, x1, y1, x2, y2, nlines);
}
