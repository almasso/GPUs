#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

__global__ void noiseReduction(uint8_t* im, float* NR, int width, int height) {
	int i, j;
	for(i=2; i < height - 2; i++) // Recorre filas
		for(j=2; j < width - 2; j++) // Recorre columnas
		{
			// Noise reduction
			NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
		}
}

__global__ void gradient(float* G, float* Gx, float* Gy, float* NR, float* phi, int width, int height) {
	int i, j;
	float PI = 3.141593;
	
	for(i=2; i<height-2; i++)
			for(j=2; j<width-2; j++)
			{
				// Intensity gradient of the image
				Gx[i*width+j] = 
					(1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
					+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
					+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
					+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
					+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


				Gy[i*width+j] = 
					((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
					+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
					+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
					+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

				G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
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
}

__global__ void edgeDetection(uint8_t* pedge, float* G, float* phi, int width, int height) {
	int i, j;

	// Edge
	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
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
}

__global__ void hysteresis(uint8_t* image_out, uint8_t* pedge, float* G, int width, int height) {
	int i, j, ii, jj;
	float lowthres, hithres;
	for(i=3; i<height-3; i++)
			for(j=3; j<width-3; j++)
			{
				image_out[i*width+j] = 0;
				if(G[i*width+j]>hithres && pedge[i*width+j])
					image_out[i*width+j] = 255;
				else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
					// check neighbours 3x3
					for (ii=-1;ii<=1; ii++)
						for (jj=-1;jj<=1; jj++)
							if (G[(i+ii)*width+j+jj]>hithres)
								image_out[i*width+j] = 255;
			}
}

void canny(uint8_t *im, uint8_t *image_out, int height, int width) 
{
	// Pasamos a memoria la imagen y un NR temporal
	float* NR, *G, *Gx, *Gy, *phi;
	uint8_t* imTmp, *pedge, *imageoutTmp;

	cudaMalloc((void**)&NR, width * height * sizeof(float));
	cudaMalloc((void**)&imTmp, width * height * sizeof(uint8_t));
	cudaMemcpy(imTmp, im, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
	noiseReduction<<<1,1>>>(imTmp, NR, width, height);
	cudaFree(imTmp);

	cudaMalloc((void**)&G, width * height * sizeof(float));
	cudaMalloc((void**)&Gx, width * height * sizeof(float));
	cudaMalloc((void**)&Gy, width * height * sizeof(float));
	cudaMalloc((void**)&phi, width * height * sizeof(float));
	gradient<<<1,1>>>(G, Gx, Gy, NR, phi, width, height);
	cudaFree(NR);
	cudaFree(Gx);
	cudaFree(Gy);

	cudaMalloc((void**)&pedge, width * height * sizeof(uint8_t));
	edgeDetection<<<1,1>>>(pedge, G, phi, width, height);
	cudaFree(phi);

	cudaMalloc((void**)imageoutTmp, width * height * sizeof(uint8_t));
	hysteresis<<<1,1>>>(imageoutTmp, pedge, G, width, height);
	cudaMemcpy(image_out, imageoutTmp, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(G);
	cudaFree(pedge);
	cudaFree(imageoutTmp);
}

void lane_assist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *x2, int *y1, int *y2, int *nlines)
{
	canny(im, imEdge, height, width);	
}
