CC=gcc
CFLAGS= -Wall -Wextra -Werror -std=c99 -pedantic -O3 -fopenmp -mavx2 -g
LDFLAGS= -lm

BIN=kmeans
OBJ=kmeans.o init.o utils.o

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

clean:
	$(RM) $(OBJ) $(BIN)
