/*
 * Часть 3.1: Базовая гибридная программа MPI + CUDA
 * Демонстрация взаимодействия MPI и CUDA
 */
#include <mpi.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_cuda(int rank, int* device_data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        printf("Hello from GPU, MPI rank: %d, thread %d\n", rank, tid);
        device_data[tid] = rank * 1000 + tid;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    printf("MPI process %d/%d initialized\n", rank, size);
    
    // Получение информации о GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("MPI rank %d: GPU %s\n", rank, prop.name);
    }
    
    // Выделение памяти на GPU
    int* device_data;
    cudaMalloc(&device_data, sizeof(int) * 1);
    
    // Запуск CUDA kernel
    hello_cuda<<<1, 1>>>(rank, device_data);
    cudaDeviceSynchronize();
    
    // Копирование данных обратно на CPU
    int host_data;
    cudaMemcpy(&host_data, device_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("MPI rank %d received from GPU: %d\n", rank, host_data);
    
    // Сбор данных со всех процессов
    int* all_data = NULL;
    if (rank == 0) {
        all_data = (int*)malloc(sizeof(int) * size);
    }
    
    MPI_Gather(&host_data, 1, MPI_INT, all_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\nCollected data from all GPUs: ");
        for (int i = 0; i < size; i++) {
            printf("%d ", all_data[i]);
        }
        printf("\n");
        free(all_data);
    }
    
    cudaFree(device_data);
    MPI_Finalize();
    return 0;
}
