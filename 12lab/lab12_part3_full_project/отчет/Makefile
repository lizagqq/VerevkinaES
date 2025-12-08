# Makefile для гибридных программ MPI + CUDA

NVCC = nvcc
MPICC = mpicc
CUDA_ARCH = sm_60
CUDA_FLAGS = -arch=$(CUDA_ARCH) -O3
MPI_INC = -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB = -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

# Targets
TARGETS = hybrid_cuda_hello hybrid_cuda_matvec

all: $(TARGETS)

hybrid_cuda_hello: hybrid_cuda_hello.cu
	$(NVCC) $(CUDA_FLAGS) $(MPI_INC) $(MPI_LIB) -o $@ $<

hybrid_cuda_matvec: hybrid_cuda_matvec.cu
	$(NVCC) $(CUDA_FLAGS) $(MPI_INC) $(MPI_LIB) -o $@ $<

clean:
	rm -f $(TARGETS) *.o

# Примечание: для реальной компиляции нужно настроить пути к MPI
# Можно использовать: mpicc --showme:compile и mpicc --showme:link

.PHONY: all clean
