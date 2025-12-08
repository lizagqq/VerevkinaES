#!/usr/bin/env python3
"""
Исходная неоптимизированная версия параллельного умножения матрицы на вектор
Демонстрирует типичные проблемы масштабируемости
"""

from mpi4py import MPI
import numpy as np
import time

def matrix_vector_multiply_original(matrix_size):
    """
    Неоптимизированная версия с поэлементными передачами
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()
    
    # Инициализация на корневом процессе
    if rank == 0:
        # Создание матрицы и вектора
        A = np.random.rand(matrix_size, matrix_size).astype(np.float64)
        x = np.random.rand(matrix_size).astype(np.float64)
        result = np.zeros(matrix_size, dtype=np.float64)
    else:
        A = None
        x = None
        result = None
    
    # Неоптимальная передача вектора x поэлементно
    if rank == 0:
        x_local = x.copy()
        for dest in range(1, size):
            for i in range(matrix_size):
                comm.send(x[i], dest=dest, tag=i)
    else:
        x_local = np.zeros(matrix_size, dtype=np.float64)
        for i in range(matrix_size):
            x_local[i] = comm.recv(source=0, tag=i)
    
    # Распределение строк матрицы
    rows_per_proc = matrix_size // size
    remainder = matrix_size % size
    
    if rank < remainder:
        local_rows = rows_per_proc + 1
        start_row = rank * local_rows
    else:
        local_rows = rows_per_proc
        start_row = rank * rows_per_proc + remainder
    
    end_row = start_row + local_rows
    
    # Неоптимальная передача строк матрицы последовательно
    if rank == 0:
        local_A = A[start_row:end_row, :].copy()
        for dest in range(1, size):
            dest_rows = rows_per_proc + (1 if dest < remainder else 0)
            dest_start = dest * (rows_per_proc + 1) if dest < remainder else dest * rows_per_proc + remainder
            dest_end = dest_start + dest_rows
            
            # Последовательная отправка каждой строки
            for row_idx in range(dest_start, dest_end):
                comm.Send(A[row_idx, :], dest=dest, tag=row_idx)
    else:
        local_A = np.zeros((local_rows, matrix_size), dtype=np.float64)
        for i in range(local_rows):
            comm.Recv(local_A[i, :], source=0, tag=start_row + i)
    
    # Локальное вычисление
    local_result = np.dot(local_A, x_local)
    
    # Неоптимальный сбор результатов поэлементно
    if rank == 0:
        result[start_row:end_row] = local_result
        for src in range(1, size):
            src_rows = rows_per_proc + (1 if src < remainder else 0)
            src_start = src * (rows_per_proc + 1) if src < remainder else src * rows_per_proc + remainder
            
            for i in range(src_rows):
                result[src_start + i] = comm.recv(source=src, tag=src_start + i)
    else:
        for i in range(local_rows):
            comm.send(local_result[i], dest=0, tag=start_row + i)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return execution_time, result if rank == 0 else None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Тестовые размеры
    test_sizes = [500, 1000, 2000]
    
    if rank == 0:
        print(f"=== Исходная неоптимизированная версия ===")
        print(f"Количество процессов: {size}")
        print(f"{'Размер':<10} {'Время (с)':<15}")
        print("-" * 25)
    
    for matrix_size in test_sizes:
        exec_time, result = matrix_vector_multiply_original(matrix_size)
        
        if rank == 0:
            print(f"{matrix_size:<10} {exec_time:<15.6f}")
    
    if rank == 0:
        print("\nЗамечания о неоптимальности:")
        print("1. Поэлементная передача вектора x")
        print("2. Последовательная передача строк матрицы")
        print("3. Поэлементный сбор результатов")
        print("4. Отсутствие использования коллективных операций")

if __name__ == "__main__":
    main()
