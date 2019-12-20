CC=gcc
CFLAGS= -Wall -Wextra -Werror -std=c99 -pedantic -O3 -g
LDFLAGS= -fopenmp -lm

BIN=kmeans
OBJ=kmeans.o

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

clean:
	$(RM) $(OBJ) $(BIN)
