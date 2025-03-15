#CC = /opt/nvidia/hpc_sdk/Linux_x86_64/2024/compilers/bin/nvcc
CC = /usr/local/cuda/bin/nvcc

EXE   = transpose 

SOURCES    = transpose.cu


OBJS    = $(SOURCES:.cu=.o)

CFLAGS     = -O3 -std=c++17

LIBS = -lm 

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS)  -o $@ $? $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu
	$(CC) $(CFLAGS) -c -o $@ $<


clean:
	rm -f $(OBJS) $(EXE)
