
CXX = g++
CC = gcc
#NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/bin/nvcc
NVCC = /usr/local/cuda/bin/nvcc

#OpenCV
CXX_OPENCV_FLAGS+=`pkg-config opencv --cflags`
LD_OPENCV_FLAGS+=`pkg-config opencv --libs`


CFLAGS=-O3 -g -I.
CUDAFLAGS =-G -g -O3 -I. 
CXXFLAGS=-O3 -g -I.

LIBS =-lpng -lm -L/usr/local/cuda/lib64 -lcudart

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
