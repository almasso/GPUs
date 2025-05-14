


CC=gcc

CFLAGS=-O3  -std=c99

LIB = -lm -lrt

#PGCC=nvc -Minfo -fast -acc -ta=nvidia
PGCC=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/compilers/bin/nvc -fast -acc -gpu=managed -Minfo=accel

NAME=jacobi

default: host gpu

host:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).host.exe $(LIB)

gpu: pgi


pgi:
	$(PGCC) $(NAME).c -o $(NAME).pgi.exe $(LIB)

clean: 
	rm -f _ *.exe
