/*
 * Часть 2.3: Гибридный метод сопряжённых градиентов
 * MPI + OpenMP реализация
 */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Гибридное умножение матрицы на вектор
void hybrid_mat_vec_mult(double* A, double* x, double* result, 
                         int local_rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < local_rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += A[i * cols + j] * x[j];
        }
    }
}

// Гибридное скалярное произведение
double hybrid_dot_product(double* a, double* b, int n) {
    double local_dot = 0.0;
    #pragma omp parallel for reduction(+:local_dot)
    for (int i = 0; i < n; i++) {
        local_dot += a[i] * b[i];
    }
    return local_dot;
}

// Векторные операции с OpenMP
void vector_copy(double* dst, double* src, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void vector_axpy(double* y, double a, double* x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] += a * x[i];
    }
}

void vector_scale(double* x, double a, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] *= a;
    }
}

int hybrid_conjugate_gradient(double* A, double* b, double* x, 
                              int local_rows, int global_cols, 
                              int max_iter, double tolerance, 
                              MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // Выделение памяти для вспомогательных векторов
    double* r = (double*)malloc(local_rows * sizeof(double));
    double* p = (double*)malloc(global_cols * sizeof(double));
    double* Ap = (double*)malloc(local_rows * sizeof(double));
    
    // Инициализация: r = b - A*x
    hybrid_mat_vec_mult(A, x, Ap, local_rows, global_cols);
    
    #pragma omp parallel for
    for (int i = 0; i < local_rows; i++) {
        r[i] = b[i] - Ap[i];
    }
    
    // p = r (глобальный)
    memset(p, 0, global_cols * sizeof(double));
    
    // Заполнение локальной части p
    int start_idx = 0;
    MPI_Scan(&local_rows, &start_idx, 1, MPI_INT, MPI_SUM, comm);
    start_idx -= local_rows;
    
    for (int i = 0; i < local_rows; i++) {
        p[start_idx + i] = r[i];
    }
    
    // Глобализация p
    MPI_Allreduce(MPI_IN_PLACE, p, global_cols, MPI_DOUBLE, MPI_SUM, comm);
    
    // r^T * r
    double local_rsold = hybrid_dot_product(r, r, local_rows);
    double rsold;
    MPI_Allreduce(&local_rsold, &rsold, 1, MPI_DOUBLE, MPI_SUM, comm);
    
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        hybrid_mat_vec_mult(A, p, Ap, local_rows, global_cols);
        
        // alpha = rsold / (p^T * Ap)
        double local_pAp = hybrid_dot_product(p + start_idx, Ap, local_rows);
        double pAp;
        MPI_Allreduce(&local_pAp, &pAp, 1, MPI_DOUBLE, MPI_SUM, comm);
        
        if (fabs(pAp) < 1e-15) {
            if (rank == 0) {
                printf("Warning: pAp too small at iteration %d\n", iter);
            }
            break;
        }
        
        double alpha = rsold / pAp;
        
        // x = x + alpha * p
        vector_axpy(x, alpha, p, global_cols);
        
        // r = r - alpha * Ap
        vector_axpy(r, -alpha, Ap, local_rows);
        
        // rsnew = r^T * r
        double local_rsnew = hybrid_dot_product(r, r, local_rows);
        double rsnew;
        MPI_Allreduce(&local_rsnew, &rsnew, 1, MPI_DOUBLE, MPI_SUM, comm);
        
        // Проверка сходимости
        if (sqrt(rsnew) < tolerance) {
            if (rank == 0) {
                printf("Converged at iteration %d, residual: %.6e\n", 
                       iter + 1, sqrt(rsnew));
            }
            iter++;
            break;
        }
        
        // beta = rsnew / rsold
        double beta = rsnew / rsold;
        
        // p = r + beta * p
        // Сначала обновляем локальную часть p
        for (int i = 0; i < local_rows; i++) {
            p[start_idx + i] = r[i] + beta * p[start_idx + i];
        }
        
        // Глобализация p
        MPI_Allreduce(MPI_IN_PLACE, p, global_cols, MPI_DOUBLE, MPI_SUM, comm);
        
        rsold = rsnew;
    }
    
    free(r);
    free(p);
    free(Ap);
    
    return iter;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Параметры задачи
    int global_rows = 2000;
    int global_cols = 2000;
    int local_rows = global_rows / size;
    
    if (rank == 0) {
        printf("======================================\n");
        printf("Hybrid Conjugate Gradient Method\n");
        printf("======================================\n");
        printf("Problem size: %d x %d\n", global_rows, global_cols);
        printf("MPI processes: %d\n", size);
        printf("Local rows per process: %d\n", local_rows);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("======================================\n");
    }
    
    // Выделение памяти
    double* A = (double*)malloc(local_rows * global_cols * sizeof(double));
    double* b = (double*)malloc(local_rows * sizeof(double));
    double* x = (double*)calloc(global_cols, sizeof(double));
    
    // Инициализация: создаём матрицу с диагональным преобладанием
    srand(rank * 1000);
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < global_cols; j++) {
            A[i * global_cols + j] = ((double)rand() / RAND_MAX) * 0.1;
        }
        // Усиление диагонали
        int global_row = rank * local_rows + i;
        if (global_row < global_cols) {
            A[i * global_cols + global_row] += 5.0;
        }
    }
    
    // Инициализация правой части
    for (int i = 0; i < local_rows; i++) {
        b[i] = (double)(rank * local_rows + i + 1) * 0.1;
    }
    
    // Решение системы
    if (rank == 0) {
        printf("\nSolving system...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    int iterations = hybrid_conjugate_gradient(A, b, x, local_rows, global_cols,
                                               100, 1e-6, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("\n======================================\n");
        printf("Execution time: %.6f seconds\n", end_time - start_time);
        printf("Iterations: %d\n", iterations);
        printf("Solution (first 5 elements): ");
        for (int i = 0; i < 5 && i < global_cols; i++) {
            printf("%.4f ", x[i]);
        }
        printf("\n======================================\n");
    }
    
    free(A);
    free(b);
    free(x);
    
    MPI_Finalize();
    return 0;
}
