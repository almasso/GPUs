// Implementación de lane-assist en GPU utilizando CUDA - Alejandro Massó Martínez
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <algorithm>
#include "png_io.h"
#include "routinesGPU.h"

#define BLOCK_SIZE 32
#define PADDING 2

__global__ void noiseReduction(uint8_t* im, float* NR, int width, int height) {
	// Declaramos tx, ty y las posiciones globales de cada hilo
	int tx = threadIdx.x, ty = threadIdx.y;
    int i = ty + blockIdx.y * blockDim.y;
    int j = tx + blockIdx.x * blockDim.x;

	// Si la posición global es mayor que la anchura o la altura, no hacemos nada
    if (i >= height || j >= width) return;

	// Declaramos una variable en memoria compartida, dejando espacio para el borde de la imagen (2 píxeles 
	// por las 4 direcciones, por tanto + 4 en cada dimensión), y con un padding de 2 para desalinear la memoria (mejora 
	// de accesos).
    __shared__ float sNR[BLOCK_SIZE + 4 + PADDING][BLOCK_SIZE + 4];

	// Hacemos que el hilo (y,x) guarde en la posición (y + 2, x + 2), explotando el acceso coalescente.
    sNR[ty + 2][tx + 2] = im[i * width + j];

    // Cargamos los marcos de 2 de ancho que va a tener la matriz. Con esto nos aseguramos que todos los píxeles tienen disponible
	// la superficie de 5x5 que necesitan para obtener el resultado final en NR
    if (ty < 2 && i >= 2) {
        sNR[ty][tx + 2] = im[(i - 2) * width + j];
	}
	if (ty < 2 && i + BLOCK_SIZE < height) {
        sNR[ty + BLOCK_SIZE + 2][tx + 2] = im[(i + BLOCK_SIZE) * width + j];
    }
    if (tx < 2 && j >= 2) {
        sNR[ty + 2][tx] = im[i * width + (j - 2)];
	}
	if (tx < 2 && j + BLOCK_SIZE < width) {
        sNR[ty + 2][tx + BLOCK_SIZE + 2] = im[i * width + (j + BLOCK_SIZE)];
    }
    __syncthreads();

	// En lugar de hacer el cálculo como está en routinesCPU.c, almacenamos los pesos
	// en una matriz 5x5 para hacer un mejor uso de los registros: en lugar de hacer
	// una operación muy grande con muchos accesos de memoria, de los cuales no todos
	// se van a poder cargar al mismo tiempo y van a tener que ir a memoria local,
	// hacemos múltiples operaciones y las almacenamos en una variable resultado.
	float weights[5][5] = {
		{2.0, 4.0, 5.0, 4.0, 2.0},
		{4.0, 9.0, 12.0, 9.0, 4.0},
		{5.0, 12.0, 15.0, 12.0, 5.0},
		{4.0, 9.0, 12.0, 9.0, 4.0},
		{2.0, 4.0, 5.0, 4.0, 2.0}
	};

    float res = 0.0f;
    for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 5; dx++) {
            res += weights[dy][dx] * sNR[ty + dy][tx + dx];
        }
    }
    res /= 159.0f;

    // Almacenamos el resultado en NR.
    NR[i * width + j] = res;
}

__global__ void gradientX(float* Gx, float* NR, int width, int height) {
	// Siguiendo la misma filosofía que en el kernel de noiseReduction, vamos a hacer lo mismo aquí,
	// con la única diferencia de los valores de la matriz de pesos.
	int tx = threadIdx.x, ty = threadIdx.y;
	int i = ty + blockIdx.y * blockDim.y;
	int j = tx + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	__shared__ float sGx[BLOCK_SIZE + 4 + PADDING][BLOCK_SIZE + 4];

	sGx[ty + 2][tx + 2] = NR[i * width + j];

	if (ty < 2 && i >= 2) {
        sGx[ty][tx + 2] = NR[(i - 2) * width + j];
	}
	if (ty < 2 && i + BLOCK_SIZE < height) {
        sGx[ty + BLOCK_SIZE + 2][tx + 2] = NR[(i + BLOCK_SIZE) * width + j];
    }
    if (tx < 2 && j >= 2) {
        sGx[ty + 2][tx] = NR[i * width + (j - 2)];
	}
	if (tx < 2 && j + BLOCK_SIZE < width) {
        sGx[ty + 2][tx + BLOCK_SIZE + 2] = NR[i * width + (j + BLOCK_SIZE)];
    }
	__syncthreads();

	// Esta matriz de pesos tiene una columna llena completamente de 0.
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
	// Siguiendo la misma filosofía que en el kernel de noiseReduction, vamos a hacer lo mismo aquí,
	// con la única diferencia de los valores de la matriz de pesos.
	int tx = threadIdx.x, ty = threadIdx.y;
	int i = ty + blockIdx.y * blockDim.y;
	int j = tx + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	__shared__ float sGy[BLOCK_SIZE + 4 + PADDING][BLOCK_SIZE + 4];

	sGy[ty + 2][tx + 2] = NR[i * width + j];

	if (ty < 2 && i >= 2) {
        sGy[ty][tx + 2] = NR[(i - 2) * width + j];
	}
	if (ty < 2 && i + BLOCK_SIZE < height) {
        sGy[ty + BLOCK_SIZE + 2][tx + 2] = NR[(i + BLOCK_SIZE) * width + j];
    }
    if (tx < 2 && j >= 2) {
        sGy[ty + 2][tx] = NR[i * width + (j - 2)];
	}
	if (tx < 2 && j + BLOCK_SIZE < width) {
        sGy[ty + 2][tx + BLOCK_SIZE + 2] = NR[i * width + (j + BLOCK_SIZE)];
    }
	__syncthreads();

	// Esta matriz de pesos tiene una fila llena completamente de 0.
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
	// En gradient simplemente vamos a almacenar un valor en el array G, por lo que este
	// kernel no utiliza apenas cómputo.
	int tx = threadIdx.x, ty = threadIdx.y;
	int i = ty + blockIdx.y * blockDim.y;
	int j = tx + blockIdx.x * blockDim.x;

	if(i >= height || j >= width) return;

	G[i*width+j] = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
}

__global__ void phiKernel(float* phi, float* Gx, float* Gy, int width, int height) {
	// Lo mismo sucede con este kernel. Simplemente se ha cambiado la estructura
	// para evitar lo máximo posible accesos a Gx y Gy que realentizan la ejecución,
	// y se ha simplificado qué valor se le da a phi para que sea más legible.
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if((i < 2 || i > height - 3) || (j < 2 || j > width - 3)) return;

	float PI = 3.141593;
	
	float gx = Gx[i * width + j];
	float gy = Gy[i * width + j];

	float angle = atan2f(fabsf(gy), fabsf(gx));

	phi[i * width + j] = ((angle <= PI/8) ? 0 :
						  (angle <= 3 * PI/8) ? 45 :
						  (angle <= 5 * PI/8) ? 90 :
						  (angle <= 7 * PI/8) ? 135 : 0);

}

__global__ void edgeDetection(uint8_t* pedge, float* G, float* phi, int width, int height) {
	// En este kernel también se ha intentado hacer más legible el valor de pedge. El valor de pedge
	// nos obliga siempre a estar accediendo a G, por lo que los accesos a memoria no se pueden optimizar
	// demasiado, solo los que tengan que ver con gVal y phiVal.
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;

	if((i < 3 || i > height - 4) || (j < 3 || j > width - 4)) return;

	float gVal = G[i * width + j];
    float phiVal = phi[i * width + j];
    
    pedge[i * width + j] = (phiVal == 0 && gVal > G[i * width + j + 1] && gVal > G[i * width + j - 1]) ||
                           (phiVal == 45 && gVal > G[(i + 1) * width + j + 1] && gVal > G[(i - 1) * width + j - 1]) ||
                           (phiVal == 90 && gVal > G[(i + 1) * width + j] && gVal > G[(i - 1) * width + j]) ||
                           (phiVal == 135 && gVal > G[(i + 1) * width + j - 1] && gVal > G[(i - 1) * width + j + 1]);
}

__global__ void hysteresis(uint8_t* image_out, uint8_t* pedge, float* G, int width, int height, float level) {
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	
	if((i < 3 || i > height - 4) || (j < 3 || j > width - 4)) return;

	float lowthres = level/2, hithres = 2 *(level);
	float gVal = G[i * width + j];

	// Distinguimos entre un borde "fuerte", si el valor del gradiente es mayor que el umbral mayor definido y es um
	// potencial borde; y borde "débil", si el valor del gradiente es menor que el umbral mayor pero mayor que el umbral
	// menor.
	bool strEdge = (gVal > hithres) && pedge[i * width + j];
	bool weakEdge = (gVal >= lowthres && gVal < hithres && pedge[i * width + j]);

	// Si el borde es "fuerte", lo ponemos a 255
	if(strEdge) {
		image_out[i * width + j] = 255;
	}

	// Si es "débil", comprobamos en su vecindad (3x3) si hubiese algún potencial borde "fuerte",
	// y lo promocionamos a ser un borde "fuerte".
	else if(weakEdge) {
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

    uint8_t pixelVal = im[i * width + j];

	// Solo nos interesan los bordes que sean "fuertes", por lo que el resto los descartamos.
    if (pixelVal <= 250) return;

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
	// En este kernel se han podido eliminar los dos bucles for anidados
	// utilizando, en lugar de i y j, rho y theta. El resto, está igual que en
	// el routinesCPU.c, excepto el uso de atomicAdd en lugar de var++.
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
	// Declaramos las variables que vamos a utilizar
	float* NR, *G, *Gx, *Gy, *phi;
	uint8_t* imTmp, *pedge, *imageoutTmp;

	// Calculamos las dimensiones del grid
	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	if(width % dimBlock.x != 0) xSum = 1;
	if(height % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((width / dimBlock.x) + xSum, (height / dimBlock.y) + ySum);

	// Usaremos streams para paralelizar algunos kernels
	int nStreams = 2;
	cudaStream_t streams[nStreams];
	for(int i = 0; i < nStreams; ++i)
		cudaStreamCreate(&streams[i]);

	// Nos guardamos una referencia a las variables que vamos a utilizar en el kernel noiseReduction
	NR = fVar;
	imTmp = cVar;
	// Copiamos la imagen a la memoria del device
	cudaMemcpy(imTmp, im, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

	// Llamamos al kernel noiseReduction
	noiseReduction<<<dimGrid,dimBlock>>>(imTmp, NR, width, height);
	cudaDeviceSynchronize();
	
	// Seteamos a nullptr las variables que ya no se vayan a usar
	imTmp = nullptr;
	
	// Nos guardamos referencia a las variables que vamos a usar en los kernel gradientX y gradientY
	Gx = fVar + (2 * width * height);
	Gy = fVar + (3 * width * height);

	// Como gradientX y gradientY únicamente requieren de leer de NR, podemos llamarlos en streams distintos para paralelizar
	gradientX<<<dimGrid, dimBlock, 0, streams[0]>>>(Gx, NR, width, height);
	gradientY<<<dimGrid, dimBlock, 0, streams[1]>>>(Gy, NR, width, height);
	
	// Sincronizamos los dos streams
	for(int i = 0; i < nStreams; ++i)
		cudaStreamSynchronize(streams[i]);
	
	// Quitamos la referencia a lo que no vayamos a usar, y nos guardamos referencia a los que vayamos a usar en los kernel gradient y phiKernel
	NR = nullptr;
	G = fVar;
	phi = fVar + (width * height);

	// Llamamos paralelamente a los kernels gradient y phiKernel, ya que los dos simplemente leen de Gx y Gy y no hacen nada entre ellos
	gradient<<<dimGrid, dimBlock, 0, streams[0]>>>(G, Gx, Gy, width, height);
	phiKernel<<<dimGrid, dimBlock, 0, streams[1]>>>(phi, Gx, Gy, width, height);
	for(int i = 0; i < nStreams; ++i)
		// Sincronizamos los kernels
		cudaStreamSynchronize(streams[i]);
	
	Gx = nullptr;
	Gy = nullptr;
	pedge = cVar;
	// De manera asíncrona, seteamos la memoria de pedge inicialmente a 0 en lugar de igualarla directamente dentro de la función, y, en el mismo
	// stream, ejecutamos el kernel de edgeDetection
	cudaMemsetAsync(pedge, 0, width * height * sizeof(uint8_t), streams[0]);
	edgeDetection<<<dimGrid, dimBlock, 0, streams[0]>>>(pedge, G, phi, width, height);
	cudaStreamSynchronize(streams[0]);

	phi = nullptr;
	imageoutTmp = cVar + (width * height);
	// De igual manera que antes, llamamos asíncronamente al memset de la imagen temporal y a la función que le va a setear un valor en B/W.
	cudaMemsetAsync(imageoutTmp, 0, width * height * sizeof(uint8_t), streams[0]);
	hysteresis<<<dimGrid, dimBlock, 0, streams[0]>>>(imageoutTmp, pedge, G, width, height, level);
	cudaStreamSynchronize(streams[0]);
	
	// Destruimos todos los streams
	for(int i = 0; i < nStreams; ++i) 
		cudaStreamDestroy(streams[i]);

	// Igualamos todas las variables a nullptr por seguridad
	G = nullptr;
	pedge = nullptr;
	imageoutTmp = nullptr;
}

void houghTransform(uint8_t* cVar, float* fVar, uint32_t* lVar, int width, int height, int accu_width, int accu_height, 
	float* sin_table, float* cos_table) 
{
	// Declaramos las variables que vamos a utilizar
	uint8_t* imTmp;
	uint32_t* accuTmp;
	float* sinTTmp, *cosTTmp;

	// Calculamos las dimensiones del grid. En este caso, el profiler de NCU decía que el tamaño de bloque utilizado no era óptimo,
	// y, reduciéndolo a la mitad, vi que los resultados mejoraban bastante, por eso en este caso el tamaño de bloque es la mitad.
	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE / 2, BLOCK_SIZE / 2);
	if(width % dimBlock.x != 0) xSum = 1;
	if(height % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((width / dimBlock.x) + xSum, (height / dimBlock.y) + ySum);

	// Nos guardamos referencia a las variables que vamos a utilizar, 
	// teniendo en cuenta dónde se han guardado las del método anterior que vamos a usar en este
	imTmp = cVar + (width * height);
	sinTTmp = fVar;
	cosTTmp = fVar + 180;
	accuTmp = lVar;

	// Copiamos en memoria del device las tablas de seno, coseno y el acumulador (inicializado a 0)
	cudaMemcpy(sinTTmp, sin_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cosTTmp, cos_table, 180 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(accuTmp, 0, accu_width * accu_height * sizeof(uint32_t));

	// Llamamos al kernel del algoritmo de Hough
	houghKernel<<<dimGrid, dimBlock>>>(imTmp, accuTmp, width, height, accu_width, accu_height, sinTTmp, cosTTmp);
	cudaDeviceSynchronize();
	
	// Por seguridad, ponemos las variables a nullptr
	imTmp = nullptr;
	sinTTmp = nullptr;
	cosTTmp = nullptr;
	accuTmp = nullptr;
}

void getLines(int threshold, uint32_t* lVar, float* fVar, int* iVar, int accu_width, int accu_height, int width, int height, 
	int* x1, int* y1, int* x2, int* y2, int* lines) 
{
	// Declaramos las variables que vamos a utilizar
	uint32_t* accuTmp;
	float* sinTTmp, *cosTTmp;
	int* x1Tmp, *y1Tmp, *x2Tmp, *y2Tmp, *linesTmp;

	// Calculamos las dimensiones del grid. En este caso, hay que tener en cuenta a accu_width y accu_height junto con width y height para ver
	// qué tamaño es mayor.
	int xSum = 0, ySum = 0;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int maximumWidth = std::max(width, accu_width);
	int maximumHeight = std::max(height, accu_height);
	if(maximumWidth % dimBlock.x != 0) xSum = 1;
	if(maximumHeight % dimBlock.y != 0) ySum = 1;
	dim3 dimGrid((maximumWidth / dimBlock.x) + xSum, (maximumHeight / dimBlock.y) + ySum);

	// Nos guardamos referencia a las variables que vamos a utilizar, 
	// teniendo en cuenta dónde se han guardado las del método anterior que vamos a usar en este
	accuTmp = lVar;
	sinTTmp = fVar;
	cosTTmp = fVar + 180;
	x1Tmp = iVar;
	y1Tmp = iVar + 10;
	x2Tmp = iVar + 20;
	y2Tmp = iVar + 30;
	linesTmp = iVar + 40;

	// Seteamos a 0 el número de líneas con el que comenzamos
	cudaMemset(linesTmp, 0, sizeof(int));
	
	// Llamamos al kernel getLines
	getLinesKernel<<<dimGrid, dimBlock>>>(threshold, accuTmp, accu_width, accu_height, width, height, sinTTmp, cosTTmp, x1Tmp, y1Tmp, x2Tmp, y2Tmp, linesTmp);
	cudaDeviceSynchronize();

	// Copiamos todos los valores de memoria del device a memoria del host
	cudaMemcpy(x1, x1Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y1, y1Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(x2, x2Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y2, y2Tmp, 10 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(lines, linesTmp, sizeof(int), cudaMemcpyDeviceToHost);

	// Por seguridad, ponemos todas las variables a nullptr
	accuTmp = nullptr;
	sinTTmp = nullptr;
	cosTTmp = nullptr;
	x1Tmp = nullptr;
	y1Tmp = nullptr;
	x2Tmp = nullptr;
	y2Tmp = nullptr;
	linesTmp = nullptr;
}

void lane_assist_GPU(uint8_t *im, int height, int width, float *sin_table, 
	float *cos_table, int accu_height, int accu_width, int *x1, int *y1, int *x2, 
	int *y2, int *nlines, float* floatVariables, uint8_t* charVariables, uint32_t* longVariables, 
	int* intVariables)
{
	// Incializamos las variables level y threshold
	float level = 1000.0f;
	int threshold = width > height ? width/6 : height / 6;

	// Llamamos a canny
	canny(im, charVariables, floatVariables, height, width, level);

	// De charVariables, nos guardamos una referencia a imEdge
	uint8_t* imEdge = charVariables + (width * height);
	uint8_t* imEdgeCPU = (uint8_t*)malloc(width * height * sizeof(uint8_t));
	// Lo pasamos a memoria del host
	cudaMemcpy(imEdgeCPU, imEdge, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	// Pintamos la imagen intermedia
	write_png_fileBW("out_edges.png", imEdgeCPU, width, height);
	// Liberamos la memoria host que hemos reservado
	free(imEdgeCPU);
	imEdgeCPU = nullptr;
	imEdge = nullptr;

	// Llamamos a houghTransform
	houghTransform(charVariables, floatVariables, longVariables, width, height, accu_width, accu_height, sin_table, cos_table);

	// Llamamos a getLines
	getLines(threshold, longVariables, floatVariables, intVariables, accu_width, accu_height, width, height, x1, y1, x2, y2, nlines);
}

void loadVariables(float** floatVariables, uint8_t** charVariables, uint32_t** longVariables, int** intVariables, int width, int height, int accu_width, int accu_height) {
	// Cargamos todas las variables al principio, dándoles un tamaño grande para evitar quedarnos sin memoria en todo momento
	// Solo se van a usar 2 variables de tipo uint8_t al mismo tiempo en los kernels, por lo que le ponemos de tamaño
	// 2 veces * tamaño de la variable más grande (que es width * height).
	cudaMalloc((void**)charVariables, 2 * width * height * sizeof(uint8_t));
	// En el caso de variables de tipo float, el máximo concurrente van a ser 4, por lo que inicializamos un array de tamaño 4 * width * height
	cudaMalloc((void**)floatVariables, 4 * width * height * sizeof(float));
	// En el caso de las variables de tipo long, solo se va a estar usando 1 en cualquier momento, por lo que solo inicializamos 
	// una de tamaño accu_width * accu_height
	cudaMalloc((void**)longVariables, accu_width * accu_height * sizeof(uint32_t));
	// En las de int, vamos a tener 4 arrays de tamaño 10 y un único número (el número de líneas totales), pero es más fácil consumir un poco más
	// de memoria, por lo que 5 * 10.
	cudaMalloc((void**)intVariables, 5 * 10 * sizeof(int));
}

void freeVariables(float* floatVariables, uint8_t* charVariables, uint32_t* longVariables, int* intVariables) {
	// Liberamos las variables inicializadas en loadVariables y las igualamos a nullptr por seguridad.
	cudaFree(charVariables);
	charVariables = nullptr;
	cudaFree(floatVariables);
	floatVariables = nullptr;
	cudaFree(longVariables);
	longVariables = nullptr;
	cudaFree(intVariables);
	intVariables = nullptr;
}
