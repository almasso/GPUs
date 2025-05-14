
CC=gcc

CFLAGS=-O3 -lm -lrt 

PGCC=nvc -fast -acc -gpu=managed -Minfo=accel


NAME=image

default: host gpu

host:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).host.exe

gpu: pgi


pgi:
	$(PGCC) $(NAME).c -o $(NAME).pgi.exe

clean: 
	rm -f _ *.exe
