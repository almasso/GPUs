
#include "kernels.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <array>
#define MAX_WINDOW_SIZE 5*5

using  namespace  cl::sycl;

void _buble_sort(float array[], int size)
{
	int i, j;
	float tmp;

	for (i=1; i<size; i++)
		for (j=0 ; j<size - i; j++)
			if (array[j] > array[j+1]){
				tmp = array[j];
				array[j] = array[j+1];
				array[j+1] = tmp;
			}
}

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{
	int ws2 = (window_size-1)>>1; 

	Q.submit([&](handler &h) {
		h.parallel_for(range<2>(height - ws2, width - ws2), [=](id<2> pixel) {
			float median;
			float window[MAX_WINDOW_SIZE];
			int i = pixel[0], j = pixel[1], ii, jj;

			if(i >= ws2 && j >= ws2) {
				for (ii =-ws2; ii<=ws2; ii++)
				for (jj =-ws2; jj<=ws2; jj++)
					window[(ii+ws2)*window_size + jj+ws2] = im[(i+ii)*width + j+jj];

			// SORT
			_buble_sort(window, window_size*window_size);
			median = window[(window_size*window_size-1)>>1];

			if (fabsf((median-im[i*width+j])/median) <=thredshold)
				image_out[i*width + j] = im[i*width+j];
			else
				image_out[i*width + j] = median;
			}

		});
	}).wait();
}
