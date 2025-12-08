/*
 * Часть 2.2: Гибридное умножение матрицы на вектор
 * MPI для распределения данных + OpenMP для многопоточности
 */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void matrix_vector_multiply(double* A, double* x, double* b, 
                            int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        b[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            b[i] += A[i * cols + j] * x[j];
        }
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Параметры задачи
    int total_rows = 10000;
    int cols = 10000;
    int rows_per_process = total_rows / size;
    
    if (rank == 0) {
        printf("======================================\n");
        printf("Hybrid Matrix-Vector Multiplication\n");
        printf("======================================\n");
        printf("Problem size: %d x %d\n", total_rows, cols);
        printf("MPI processes: %d\n", size);
        printf("Rows per process: %d\n", rows_per_process);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("======================================\n");
    }
    
    // Выделение памяти
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
    
    // Широковещательная рассылка вектора x
    MPI_Bcast(local_x, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Синхронизация и измерение времени
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    // Гибридное вычисление (MPI + OpenMP)
    matrix_vector_multiply(local_A, local_x, local_b, rows_per_process, cols);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("\nExecution time: %.6f seconds\n", end_time - start_time);
        printf("Result (first 3 elements): %.4f, %.4f, %.4f\n", 
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
