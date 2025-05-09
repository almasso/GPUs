#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "routinesCPU.h"
#include "routinesGPU.h"
#include "png_io.h"

/* Time */
#include <sys/time.h>
#include <sys/resource.h>


static struct timeval tv0;
double get_time()
{
	double t;
	gettimeofday(&tv0, (struct timezone*)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);

	return (t);
}

 

int main(int argc, char **argv)
{
	uint8_t *imtmp, *im;
	int width, height;
	bool verbose = true;

	float sin_table[180], cos_table[180];
	int nlines=0; 
	int x1[10], x2[10], y1[10], y2[10];
	int l;
	double t0, t1;


	/* Only accept a concrete number of arguments */
	if(argc < 3 || argc > 4)
	{
		printf("./exec image.png [c/g] [t/v (def v)]\n");
		exit(-1);
	}

	if(argc == 4) {
		switch(argv[3][0]) {
			case 't':
				verbose = false;
				break;
			case 'v':
				verbose = true;
				break;
			default:
				verbose = true;
				break;
		}
	}

	/* Read images */
	imtmp = read_png_fileRGB(argv[1], &width, &height);
	im    = image_RGB2BW(imtmp, height, width);

	init_cos_sin_table(sin_table, cos_table, 180);	

	// Create temporal buffers 
	uint8_t *imEdge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float *NR = (float *)malloc(sizeof(float) * width * height);
	float *G = (float *)malloc(sizeof(float) * width * height);
	float *phi = (float *)malloc(sizeof(float) * width * height);
	float *Gx = (float *)malloc(sizeof(float) * width * height);
	float *Gy = (float *)malloc(sizeof(float) * width * height);
	uint8_t *pedge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	//Create the accumulators
	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
	int accu_height = hough_h * 2.0; // -rho -> +rho
	int accu_width  = 180;
	uint32_t *accum = (uint32_t*)malloc(accu_width*accu_height*sizeof(uint32_t));

	// Variables para la llamada de GPU
	// Con estas cuatro variables tendremos suficiente y solo tendremos que llamar 
	// 4 veces a cudaMalloc y cudaFree, aumentando bastante el rendimiento
	float* fVar = nullptr; // Almacenará todas las variables de tipo float que se usen en los kernels
	uint32_t* lVar = nullptr; // Almacenará todas las variables de tipo uint32_t que se usen en los kernels
	uint8_t* cVar = nullptr; // Almacenará todas las variables de tipo uint8_t que se usen en los kernels
	int* iVar = nullptr; // Almacenará todas las variables de tipo int (posiblemente, uint16_t) que se usen en los kernels
	
	switch (argv[2][0]) {
		case 'c':
			t0 = get_time();
			lane_assist_CPU(im, height, width, 
				imEdge, NR, G, phi, Gx, Gy, pedge,
				sin_table, cos_table,
				accum, accu_height, accu_width,
				x1, y1, x2, y2, &nlines);
			t1 = get_time();

			if(verbose) 
				printf("CPU Exection time %f \xC2\xB5s.\n", t1-t0);
			else
				printf("%f\n", t1 - t0);
			break;
		case 'g':
			// Primero cargamos las variables, para que el tiempo de carga no se tenga en cuenta en el tiempo que tarda el kernel
			loadVariables(&fVar, &cVar, &lVar, &iVar, width, height, accu_width, accu_height);
			t0 = get_time();
			// Llamada a lane_assist_GPU después de inicializar las variables
			lane_assist_GPU(im, height, width, sin_table, cos_table, accu_height, accu_width,
				x1, y1, x2, y2, &nlines, fVar, cVar, lVar, iVar);
			t1 = get_time();
			// Por último, liberamos las variables, para que tampoco se tenga su tiempo en cuenta en el tiempo de ejecución del kernel
			freeVariables(fVar, cVar, lVar, iVar);
			if(verbose)
				printf("GPU Exection time %f \xC2\xB5s.\n", t1-t0);
			else
				printf("%f\n", t1 - t0);
			break;
		default:
			printf("Not Implemented yet!!\n");
	}

	if(verbose)
		for (int l=0; l<std::min(nlines, 10); l++)
			printf("(x1,y1)=(%d,%d) (x2,y2)=(%d,%d)\n", x1[l], y1[l], x2[l], y2[l]);

	draw_lines(imtmp, width, height, x1, y1, x2, y2, nlines);

	write_png_fileRGB("out.png", imtmp, width, height);
}
