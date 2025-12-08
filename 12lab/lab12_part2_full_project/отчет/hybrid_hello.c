/*
 * Часть 2.1: Базовая гибридная программа MPI + OpenMP
 * Демонстрация работы MPI и OpenMP
 */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Проверка уровня поддержки потоков
    if (world_rank == 0) {
        printf("======================================\n");
        printf("Hybrid MPI+OpenMP Hello World\n");
        printf("======================================\n");
        printf("MPI processes: %d\n", world_size);
        printf("MPI thread support: ");
        switch(provided) {
            case MPI_THREAD_SINGLE:     printf("SINGLE\n"); break;
            case MPI_THREAD_FUNNELED:   printf("FUNNELED\n"); break;
            case MPI_THREAD_SERIALIZED: printf("SERIALIZED\n"); break;
            case MPI_THREAD_MULTIPLE:   printf("MULTIPLE\n"); break;
        }
        printf("======================================\n\n");
    }
    
    // Параллельная область OpenMP
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        // Критическая секция для упорядоченного вывода
        #pragma omp critical
        {
            printf("MPI process %d/%d, OpenMP thread %d/%d on host\n", 
                   world_rank, world_size, thread_id, num_threads);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        printf("\n======================================\n");
        printf("Hybrid execution completed\n");
        printf("======================================\n");
    }
    
    MPI_Finalize();
    return 0;
}
