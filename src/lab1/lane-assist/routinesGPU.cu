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

    if (i >= height || j >= width) return;

    __shared__ float sNR[BLOCK_SIZE + 4][BLOCK_SIZE + 4 + 2]; // Usamos un padding de 2

    sNR[ty + 2][tx + 2] = im[i * width + j];

    // Cargamos los marcos de 2 de ancho que va a tener la matriz
    if (ty < 2) {
        sNR[ty][tx + 2] = im[(i - 2) * width + j];
        sNR[ty + BLOCK_SIZE + 2][tx + 2] = im[(i + BLOCK_SIZE) * width + j];
    }
    if (tx < 2) {
        sNR[ty + 2][tx] = im[i * width + (j - 2)];
        sNR[ty + 2][tx + BLOCK_SIZE + 2] = im[i * width + (j + BLOCK_SIZE)];
    }
    __syncthreads();

	// Matriz de pesos
	float weights[5][5] = {
		{2.0, 4.0, 5.0, 4.0, 2.0},
		{4.0, 9.0, 12.0, 9.0, 4.0},
		{5.0, 12.0, 15.0, 12.0, 5.0},
		{4.0, 9.0, 12.0, 9.0, 4.0},
		{2.0, 4.0, 5.0, 4.0, 2.0}
	};

    // Hacemos el cálculo de la matriz NR
    float res = 0.0f;
    for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 5; dx++) {
            res += weights[dy][dx] * sNR[ty + dy][tx + dx];
        }
    }
    res /= 159.0f;

    // Almacenamos el resultado
    NR[i * width + j] = res;
}

__global__ void gradientX(float* Gx, float* NR, int width, int height) {
	int tx = threadIdx.x, ty = threadIdx.y;
	int i = ty + blockIdx.y * blockDim.y;
	int j = tx + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	__shared__ float sGx[BLOCK_SIZE + 4][BLOCK_SIZE + 4 + 2]; // Usamos un padding de 2

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

	// Matriz de pesos
	float weights[5][5] = {
		{1.0, 2.0, 0.0, -2.0, -1.0},
		{4.0, 8.0, 0.0, -8.0, -4.0},
		{6.0, 12.0, 0.0, -12.0, -6.0},
		{4.0, 8.0, 0.0, -8.0, -4.0},
		{1.0, 2.0, 0.0, -2.0, -1.0}
	};

	float res = 0.0f;
	for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 5; dx++) {
            res += weights[dy][dx] * sGx[ty + dy][tx + dx];
        }
    }
	
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

	float weights[5][5] = {
		{-1.0, -4.0, -6.0, -4.0, -1.0},
		{-2.0, -8.0, -12.0, -8.0, -2.0},
		{0.0, 0.0, 0.0, 0.0, 0.0},
		{2.0, 8.0, 12.0, 8.0, 2.0},
		{1.0, 4.0, 6.0, 4.0, 1.0}
	};

	float res = 0.0f;
	for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 5; dx++) {
            res += weights[dy][dx] * sGy[ty + dy][tx + dx];
        }
    }

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
	
	float gx = Gx[i * width + j];
	float gy = Gy[i * width + j];
	phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

	float angle = atan2f(fabsf(gy), fabsf(gx));

	phi[i * width + j] = ((angle <= PI/8) ? 0 :
						  (angle <= 3 * PI/8) ? 45 :
						  (angle <= 5 * PI/8) ? 90 :
						  (angle <= 7 * PI/8) ? 135 : 0);

}

__global__ void edgeDetection(uint8_t* pedge, float* G, float* phi, int width, int height) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if((i < 3 || i > height - 4) || (j < 3 || j > width - 4)) return;

	float g_val = G[i * width + j];
    float phi_val = phi[i * width + j];
    
    pedge[i * width + j] = (phi_val == 0 && g_val > G[i * width + j + 1] && g_val > G[i * width + j - 1]) ||
                           (phi_val == 45 && g_val > G[(i + 1) * width + j + 1] && g_val > G[(i - 1) * width + j - 1]) ||
                           (phi_val == 90 && g_val > G[(i + 1) * width + j] && g_val > G[(i - 1) * width + j]) ||
                           (phi_val == 135 && g_val > G[(i + 1) * width + j - 1] && g_val > G[(i - 1) * width + j + 1]);
}

__global__ void hysteresis(uint8_t* image_out, uint8_t* pedge, float* G, int width, int height, float level) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	
	if((i < 3 || i > height - 4) || (j < 3 || j > width - 4)) return;

	float lowthres = level/2, hithres = 2 *(level);
	float gVal = G[i * width + j];

	bool strCandidate = (gVal > hithres) && pedge[i * width + j];
	bool weakCandidate = (gVal >= lowthres && gVal < hithres && pedge[i * width + j]);

	if(strCandidate) {
		image_out[i * width + j] = 255;
	}

	else if(weakCandidate) {
		// check neighbours 3x3
		for (int ii=-1;ii<=1; ii++)
			for (int jj=-1;jj<=1; jj++)
				if (G[(i+ii)*width+j+jj]>hithres) {
					image_out[i*width+j] = 255;
					return;
				}
	}
	
}

__global__ void houghKernel(uint8_t* im, uint32_t* accumulators, int width, int height, int accu_width, int accu_height, 
	float* sin_table, float* cos_table) 
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= height || j >= width) return;

    uint8_t pixel_value = im[i * width + j];

    if (pixel_value <= 250) return;

    float hough_h = (sqrtf(2.0f) * (float)(height > width ? height : width)) / 2.0f;
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;

    for (int theta = 0; theta < 180; theta++) {
        float rho = ((j - center_x) * cos_table[theta]) + ((i - center_y) * sin_table[theta]);
        int rho_index = (int)(roundf(rho + hough_h) * 180.0f) + theta;

        atomicAdd(&accumulators[rho_index], 1);
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

void canny(uint8_t *im, uint8_t *cVar, float* fVar, int height, int width, float level) 
{
	float* NR, *G, *Gx, *Gy, *phi;
	uint8_t* imTmp, *pedge, *imageoutTmp;

	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	if(width % dimBlock.x != 0) xSum = 1;
	if(height % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((width / dimBlock.x) + xSum, (height / dimBlock.y) + ySum);

	int nStreams = 2;
	cudaStream_t streams[nStreams];
	for(int i = 0; i < nStreams; ++i)
		cudaStreamCreate(&streams[i]);

	// Pasamos a memoria la imagen y un NR temporal
	NR = fVar;
	imTmp = cVar;

	cudaMemcpy(imTmp, im, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

	noiseReduction<<<dimGrid,dimBlock>>>(imTmp, NR, width, height);
	cudaDeviceSynchronize();
	// En cuanto dejemos de usar la imagen, la liberamos de memoria
	
	// Reservamos memoria para el resto de variables que vamos a utilizar
	Gx = fVar + (2 * width * height);
	Gy = fVar + (3 * width * height);

	gradientX<<<dimGrid, dimBlock, 0, streams[0]>>>(Gx, NR, width, height);
	gradientY<<<dimGrid, dimBlock, 0, streams[1]>>>(Gy, NR, width, height);
	
	for(int i = 0; i < nStreams; ++i)
		cudaStreamSynchronize(streams[i]);
	
	G = fVar;
	phi = fVar + (width * height);

	gradient<<<dimGrid, dimBlock, 0, streams[0]>>>(G, Gx, Gy, width, height);
	phiKernel<<<dimGrid, dimBlock, 0, streams[1]>>>(phi, Gx, Gy, width, height);
	for(int i = 0; i < nStreams; ++i)
		cudaStreamSynchronize(streams[i]);
	
	// Reservamos el pedge
	pedge = cVar;
	cudaMemsetAsync(pedge, 0, width * height * sizeof(uint8_t), streams[0]);
	edgeDetection<<<dimGrid, dimBlock, 0, streams[0]>>>(pedge, G, phi, width, height);
	cudaStreamSynchronize(streams[0]);

	// Reservamos memoria para la imagen final
	imageoutTmp = cVar + (width * height);
	cudaMemsetAsync(imageoutTmp, 0, width * height * sizeof(uint8_t), streams[0]);
	hysteresis<<<dimGrid, dimBlock, 0, streams[0]>>>(imageoutTmp, pedge, G, width, height, level);
	cudaStreamSynchronize(streams[0]);
	// Y la pasamos a memoria física una vez la hayamos calculado.
	
	for(int i = 0; i < nStreams; ++i) 
		cudaStreamDestroy(streams[i]);
}

void houghTransform(uint8_t* cVar, float* fVar, uint32_t* lVar, int width, int height, int accu_width, int accu_height, 
	float* sin_table, float* cos_table) 
{
	uint8_t* imTmp;
	uint32_t* accuTmp;
	float* sinTTmp, *cosTTmp;

	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE / 2, BLOCK_SIZE / 2);
	if(width % dimBlock.x != 0) xSum = 1;
	if(height % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((width / dimBlock.x) + xSum, (height / dimBlock.y) + ySum);

	imTmp = cVar + (width * height);
	sinTTmp = fVar;
	cosTTmp = fVar + 180;
	accuTmp = lVar;

	cudaMemcpy(sinTTmp, sin_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cosTTmp, cos_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(accuTmp, 0, accu_width * accu_height * sizeof(uint32_t));

	houghKernel<<<dimGrid, dimBlock>>>(imTmp, accuTmp, width, height, accu_width, accu_height, sinTTmp, cosTTmp);
	cudaDeviceSynchronize();
}

void getLines(int threshold, uint32_t* lVar, float* fVar, int* iVar, int accu_width, int accu_height, int width, int height, 
	int* x1, int* y1, int* x2, int* y2, int* lines) 
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

	accuTmp = lVar;
	sinTTmp = fVar;
	cosTTmp = fVar + 180;
	x1Tmp = iVar;
	y1Tmp = iVar + 10;
	x2Tmp = iVar + 20;
	y2Tmp = iVar + 30;
	linesTmp = iVar + 40;

	cudaMemcpy(linesTmp, lines, sizeof(int), cudaMemcpyHostToDevice);
	
	getLinesKernel<<<dimGrid, dimBlock>>>(threshold, accuTmp, accu_width, accu_height, width, height, sinTTmp, cosTTmp, x1Tmp, y1Tmp, x2Tmp, y2Tmp, linesTmp);
	cudaDeviceSynchronize();

	cudaMemcpy(x1, x1Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y1, y1Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(x2, x2Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y2, y2Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(lines, linesTmp, sizeof(int), cudaMemcpyDeviceToHost);
}

void lane_assist_GPU(uint8_t *im, int height, int width, float *sin_table, 
	float *cos_table, int accu_height, int accu_width, int *x1, int *y1, int *x2, 
	int *y2, int *nlines, float* floatVariables, uint8_t* charVariables, uint32_t* longVariables, 
	int* intVariables)
{
	float level = 1000.0f;
	int threshold = width > height ? width/6 : height / 6;

	canny(im, charVariables, floatVariables, height, width, level);
	uint8_t* imEdge = charVariables + (width * height);
	uint8_t* imEdgeCPU = (uint8_t*)malloc(width * height * sizeof(uint8_t));
	cudaMemcpy(imEdgeCPU, imEdge, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	write_png_fileBW("out_edges.png", imEdgeCPU, width, height);
	free(imEdgeCPU);
	imEdgeCPU = nullptr;

	houghTransform(charVariables, floatVariables, longVariables, width, height, accu_width, accu_height, sin_table, cos_table);

	getLines(threshold, longVariables, floatVariables, intVariables, accu_width, accu_height, width, height, x1, y1, x2, y2, nlines);
}

void loadVariables(float** floatVariables, uint8_t** charVariables, uint32_t** longVariables, int** intVariables, int width, int height, int accu_width, int accu_height) {
	cudaMalloc((void**)charVariables, 2 * width * height * sizeof(uint8_t));
	cudaMalloc((void**)floatVariables, 4 * width * height * sizeof(float));
	cudaMalloc((void**)longVariables, accu_width * accu_height * sizeof(uint32_t));
	cudaMalloc((void**)intVariables, 5 * 10 * sizeof(int));
}

void freeVariables(float* floatVariables, uint8_t* charVariables, uint32_t* longVariables, int* intVariables) {
	cudaFree(charVariables);
	cudaFree(floatVariables);
	cudaFree(longVariables);
	cudaFree(intVariables);
}
