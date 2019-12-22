CC=gcc
CFLAGS= -Wall -Wextra -Werror -std=c99 -pedantic -O3 -fopenmp -mavx -g
LDFLAGS= -lm

BIN=kmeans
OBJ=kmeans.o utils.o

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

clean:
	$(RM) $(OBJ) $(BIN)
