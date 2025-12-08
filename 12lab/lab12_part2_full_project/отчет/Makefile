# Makefile для гибридных программ MPI + OpenMP

CC = mpicc
CFLAGS = -fopenmp -O3 -Wall -std=c99
LDFLAGS = -fopenmp -lm

# Targets
TARGETS = hybrid_hello hybrid_matvec hybrid_cg

all: $(TARGETS)

hybrid_hello: hybrid_hello.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

hybrid_matvec: hybrid_matvec.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

hybrid_cg: hybrid_cg.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

test: all
	@echo "Testing hybrid_hello..."
	export OMP_NUM_THREADS=4 && mpiexec -n 2 ./hybrid_hello
	@echo ""
	@echo "Testing hybrid_matvec..."
	export OMP_NUM_THREADS=4 && mpiexec -n 2 ./hybrid_matvec
	@echo ""
	@echo "Testing hybrid_cg..."
	export OMP_NUM_THREADS=4 && mpiexec -n 2 ./hybrid_cg

.PHONY: all clean test
