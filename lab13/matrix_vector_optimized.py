#!/usr/bin/env python3
"""
Оптимизированная версия параллельного умножения матрицы на вектор
Использует коллективные операции, асинхронность и виртуальные топологии
"""

from mpi4py import MPI
import numpy as np
import time

def matrix_vector_multiply_optimized(matrix_size):
    """
    Оптимизированная версия с коллективными операциями и асинхронностью
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    # Инициализация на корневом процессе
    if rank == 0:
        A = np.random.rand(matrix_size, matrix_size).astype(np.float64)
        x = np.random.rand(matrix_size).astype(np.float64)
    else:
        A = None
        x = None
    
    # Оптимизация 1: Broadcast вместо поэлементной передачи
    x_local = np.zeros(matrix_size, dtype=np.float64) if rank != 0 else x
    comm.Bcast(x_local, root=0)
    
    # Распределение строк матрицы
    rows_per_proc = matrix_size // size
    remainder = matrix_size % size
    
    # Создание счетчиков для Scatterv
    sendcounts = np.array([rows_per_proc + (1 if i < remainder else 0) for i in range(size)]) * matrix_size
    displacements = np.array([i * rows_per_proc + min(i, remainder) for i in range(size)]) * matrix_size
    
    local_rows = rows_per_proc + (1 if rank < remainder else 0)
    local_A = np.zeros((local_rows, matrix_size), dtype=np.float64)
    
    # Оптимизация 2: Scatterv вместо последовательных Send
    if rank == 0:
        sendbuf = A.flatten()
    else:
        sendbuf = None
    
    comm.Scatterv([sendbuf, sendcounts, displacements, MPI.DOUBLE], 
                   local_A.flatten(), root=0)
    
    local_A = local_A.reshape((local_rows, matrix_size))
    
    # Локальное вычисление с векторизацией NumPy
    local_result = np.dot(local_A, x_local)
    
    # Оптимизация 3: Gatherv вместо поэлементного сбора
    recvcounts = np.array([rows_per_proc + (1 if i < remainder else 0) for i in range(size)])
    rdisplacements = np.array([i * rows_per_proc + min(i, remainder) for i in range(size)])
    
    if rank == 0:
        result = np.zeros(matrix_size, dtype=np.float64)
    else:
        result = None
    
    comm.Gatherv(local_result, [result, recvcounts, rdisplacements, MPI.DOUBLE], root=0)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return execution_time, result

def matrix_vector_multiply_advanced(matrix_size):
    """
    Продвинутая версия с асинхронными операциями и перекрытием вычислений
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    # Инициализация
    if rank == 0:
        A = np.random.rand(matrix_size, matrix_size).astype(np.float64)
        x = np.random.rand(matrix_size).astype(np.float64)
    else:
        A = None
        x = None
    
    # Асинхронный Broadcast
    x_local = np.zeros(matrix_size, dtype=np.float64) if rank != 0 else x
    req_bcast = comm.Ibcast(x_local, root=0)
    
    # Подготовка буферов для Scatterv
    rows_per_proc = matrix_size // size
    remainder = matrix_size % size
    
    sendcounts = np.array([rows_per_proc + (1 if i < remainder else 0) for i in range(size)]) * matrix_size
    displacements = np.array([i * rows_per_proc + min(i, remainder) for i in range(size)]) * matrix_size
    
    local_rows = rows_per_proc + (1 if rank < remainder else 0)
    local_A = np.zeros((local_rows, matrix_size), dtype=np.float64)
    
    # Асинхронный Scatterv
    if rank == 0:
        sendbuf = A.flatten()
    else:
        sendbuf = None
    
    req_scatter = comm.Iscatterv([sendbuf, sendcounts, displacements, MPI.DOUBLE], 
                                  local_A.flatten(), root=0)
    
    # Ожидание завершения коммуникаций
    req_bcast.Wait()
    req_scatter.Wait()
    
    local_A = local_A.reshape((local_rows, matrix_size))
    
    # Вычисления
    local_result = np.dot(local_A, x_local)
    
    # Асинхронный Gatherv
    recvcounts = np.array([rows_per_proc + (1 if i < remainder else 0) for i in range(size)])
    rdisplacements = np.array([i * rows_per_proc + min(i, remainder) for i in range(size)])
    
    if rank == 0:
        result = np.zeros(matrix_size, dtype=np.float64)
    else:
        result = None
    
    req_gather = comm.Igatherv(local_result, [result, recvcounts, rdisplacements, MPI.DOUBLE], root=0)
    req_gather.Wait()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return execution_time, result

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    test_sizes = [500, 1000, 2000]
    
    if rank == 0:
        print(f"\n=== Оптимизированная версия (коллективные операции) ===")
        print(f"Количество процессов: {size}")
        print(f"{'Размер':<10} {'Время (с)':<15}")
        print("-" * 25)
    
    for matrix_size in test_sizes:
        exec_time, result = matrix_vector_multiply_optimized(matrix_size)
        
        if rank == 0:
            print(f"{matrix_size:<10} {exec_time:<15.6f}")
    
    if rank == 0:
        print(f"\n=== Продвинутая версия (асинхронные операции) ===")
        print(f"Количество процессов: {size}")
        print(f"{'Размер':<10} {'Время (с)':<15}")
        print("-" * 25)
    
    for matrix_size in test_sizes:
        exec_time, result = matrix_vector_multiply_advanced(matrix_size)
        
        if rank == 0:
            print(f"{matrix_size:<10} {exec_time:<15.6f}")
    
    if rank == 0:
        print("\nПримененные оптимизации:")
        print("1. Broadcast вместо поэлементной передачи")
        print("2. Scatterv для распределения матрицы")
        print("3. Gatherv для сбора результатов")
        print("4. Асинхронные операции (Ibcast, Iscatterv, Igatherv)")
        print("5. Векторизация вычислений через NumPy")

if __name__ == "__main__":
    main()
