
CXX = g++
CC = gcc
NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/bin/nvcc
#NVCC = /usr/local/cuda/bin/nvcc # He utilizado esta ruta de NVCC en mi ordenador en casa. Dejo por defecto activada la ruta de NVCC de los laboratorios.
#NVCC = nvcc

#OpenCV
CXX_OPENCV_FLAGS+=`pkg-config opencv --cflags`
LD_OPENCV_FLAGS+=`pkg-config opencv --libs`


CFLAGS=-O3 -g -I.
CUDAFLAGS =-O3 -G -g -I. 
CXXFLAGS=-O3 -g -I.

LIBS =-lpng -lm -lcuda -lcudart

SRC = png_io.o routinesCPU.o routinesGPU.o main.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

image: $(SRC) 
	$(CXX) -o image  $(SRC) $(CXXFLAGS) $(LIBS) 

clean:
	rm -f *.o image
	rm -f out.png out_edges.png
