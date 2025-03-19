#ifndef ROUTINESGPU_H
#define ROUTINESGPU_H

#include <stdint.h>

void loadVariables(float** floatVariables, uint8_t** charVariables, uint32_t** longVariables, int** intVariables, int width, int height, 
	int accu_width, int accu_height);

void freeVariables(float* floatVariables, uint8_t* charVariables, uint32_t* longVariables, int* intVariables);

void lane_assist_GPU(uint8_t *im, int height, int width, float *sin_table, 
	float *cos_table, int accu_height, int accu_width, int *x1, int *y1, int *x2, 
	int *y2, int *nlines, float* floatVariables, uint8_t* charVariables, uint32_t* longVariables, 
	int* intVariables);

#endif

