


CC=gcc

CFLAGS=-O3  -std=c99

LIB = -lm -lrt

PGCC=nvc -Minfo -fast -acc -ta=nvidia

NAME=jacobi

default: host gpu

host:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).host.exe $(LIB)

gpu: pgi


pgi:
	$(PGCC) $(NAME).c -o $(NAME).pgi.exe $(LIB)

clean: 
	rm -f _ *.exe
