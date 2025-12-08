#!/usr/bin/env python3
"""
Часть 1: Гибридная программа умножения матрицы на вектор
MPI + многопоточный NumPy (OpenMP через BLAS/LAPACK)
"""
from mpi4py import MPI
import numpy as np
import os
import time

def setup_threading(num_threads):
    """Настройка числа потоков для NumPy/OpenMP"""
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

def hybrid_matvec(nodes, threads_per_node):
    """
    Гибридное умножение матрицы на вектор
    nodes: число узлов (MPI процессов)
    threads_per_node: число потоков OpenMP на узел
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Настройка потоков
    setup_threading(threads_per_node)
    
    # Параметры задачи
    total_rows = 10000
    cols = 10000
    rows_per_process = total_rows // size
    
    if rank == 0:
        print(f"Конфигурация: {size} узлов x {threads_per_node} потоков")
        print(f"Размер задачи: {total_rows} x {cols}")
        print(f"Строк на процесс: {rows_per_process}")
    
    # Генерация локальных данных
    np.random.seed(rank)
    local_A = np.random.rand(rows_per_process, cols).astype(np.float64)
    local_x = np.random.rand(cols).astype(np.float64)
    
    # Широковещательная рассылка вектора x
    comm.Bcast(local_x, root=0)
    
    # Синхронизация и измерение времени
    comm.Barrier()
    start_time = MPI.Wtime()
    
    # Умножение матрицы на вектор (многопоточное через NumPy)
    local_b = np.dot(local_A, local_x)
    
    comm.Barrier()
    end_time = MPI.Wtime()
    
    elapsed = end_time - start_time
    
    if rank == 0:
        print(f"Время выполнения: {elapsed:.6f} сек")
        print(f"Результат: первые 3 элемента = {local_b[:3]}")
    
    return elapsed

def main():
    """Тестирование на различных конфигурациях"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print("ГИБРИДНОЕ УМНОЖЕНИЕ МАТРИЦЫ НА ВЕКТОР (MPI + OpenMP)")
        print("="*70)
    
    # Тестирование с разным числом потоков
    thread_configs = [1, 2, 4, 8]
    results = {}
    
    for threads in thread_configs:
        if rank == 0:
            print(f"\n--- Тест с {threads} потоками ---")
        
        elapsed = hybrid_matvec(size, threads)
        results[threads] = elapsed
        
        time.sleep(0.5)  # Пауза между тестами
    
    if rank == 0:
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТЫ")
        print("="*70)
        print(f"{'Потоки':<10} {'Время (сек)':<15}")
        print("-"*70)
        for threads, elapsed in results.items():
            print(f"{threads:<10} {elapsed:<15.6f}")
    
    return results

if __name__ == "__main__":
    main()
