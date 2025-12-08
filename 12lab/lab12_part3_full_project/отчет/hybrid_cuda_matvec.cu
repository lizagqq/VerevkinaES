/*
 * Часть 3.2: Гибридное умножение матрицы на вектор (MPI + CUDA)
 * Для оценки "ХОРОШО"
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel для умножения матрицы на вектор
__global__ void mat_vec_mult_kernel(double* A, double* x, double* result, 
                                     int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        double sum = 0.0;
        for (int col = 0; col < cols; col++) {
            sum += A[row * cols + col] * x[col];
        }
        result[row] = sum;
    }
}

void hybrid_mat_vec_mult(double* A, double* x, double* result, 
                         int local_rows, int cols) {
    double *d_A, *d_x, *d_result;
    
    // Выделение памяти на GPU
    cudaMalloc(&d_A, local_rows * cols * sizeof(double));
    cudaMalloc(&d_x, cols * sizeof(double));
    cudaMalloc(&d_result, local_rows * sizeof(double));
    
    // Копирование данных на GPU
    cudaMemcpy(d_A, A, local_rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, cols * sizeof(double), cudaMemcpyHostToDevice);
    
    // Запуск kernel
    int block_size = 256;
    int grid_size = (local_rows + block_size - 1) / block_size;
    
    mat_vec_mult_kernel<<<grid_size, block_size>>>(d_A, d_x, d_result, local_rows, cols);
    
    // Синхронизация
    cudaDeviceSynchronize();
    
    // Копирование результата обратно
    cudaMemcpy(result, d_result, local_rows * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_result);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Параметры задачи
    int total_rows = 10000;
    int cols = 10000;
    int rows_per_process = total_rows / size;
    
    if (rank == 0) {
        printf("======================================\n");
        printf("Hybrid MPI+CUDA Matrix-Vector Multiplication\n");
        printf("======================================\n");
        printf("Problem size: %d x %d\n", total_rows, cols);
        printf("MPI processes: %d\n", size);
        printf("Rows per process: %d\n", rows_per_process);
        printf("======================================\n");
    }
    
    // Выделение памяти для локальной части
    double* local_A = (double*)malloc(rows_per_process * cols * sizeof(double));
    double* local_x = (double*)malloc(cols * sizeof(double));
    double* local_b = (double*)malloc(rows_per_process * sizeof(double));
    
    // Инициализация данных
    for (int i = 0; i < rows_per_process * cols; i++) {
        local_A[i] = (double)(rank * rows_per_process + i / cols + 1) * 0.001;
    }
    for (int i = 0; i < cols; i++) {
        local_x[i] = 1.0;
    }
    
    // Рассылка вектора x всем процессам
    MPI_Bcast(local_x, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Синхронизация и измерение времени
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    // Гибридное вычисление на GPU
    hybrid_mat_vec_mult(local_A, local_x, local_b, rows_per_process, cols);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("\nMPI+CUDA execution time: %.6f seconds\n", end_time - start_time);
        printf("Result (first 3 elements): %.6f, %.6f, %.6f\n", 
               local_b[0], local_b[1], local_b[2]);
        printf("======================================\n");
    }
    
    // Освобождение памяти
    free(local_A);
    free(local_x);
    free(local_b);
    
    MPI_Finalize();
    return 0;
}
