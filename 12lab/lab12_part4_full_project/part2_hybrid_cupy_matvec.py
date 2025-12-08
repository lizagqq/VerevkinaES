#!/usr/bin/env python3
"""
Часть 4.2: Гибридное умножение матрицы на вектор (MPI + CuPy симуляция)
"""
from mpi4py import MPI
import numpy as np
# import cupy as cp  # В реальности
import time

def generate_data(rank, rows_per_process, cols):
    """Генерация тестовых данных"""
    np.random.seed(rank * 100)
    host_A = np.random.rand(rows_per_process, cols).astype(np.float32)
    host_x = np.random.rand(cols).astype(np.float32)
    
    # В реальности: device_A = cp.asarray(host_A)
    device_A = host_A
    device_x = host_x
    
    return device_A, device_x, host_A, host_x

def hybrid_mat_vec_mult(device_A, device_x):
    """
    Умножение матрицы на вектор (симуляция GPU)
    В реальности: return cp.dot(device_A, device_x)
    """
    return np.dot(device_A, device_x)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Параметры задачи
    total_rows = 10000
    cols = 10000
    rows_per_process = total_rows // size
    
    if rank == 0:
        print("="*70)
        print("HYBRID MPI+CUPY MATRIX-VECTOR MULTIPLICATION")
        print("="*70)
        print(f"Problem size: {total_rows} x {cols}")
        print(f"MPI processes: {size}")
        print(f"Rows per process: {rows_per_process}")
        print("="*70)
    
    # Генерация данных
    device_A, device_x, host_A, host_x = generate_data(rank, rows_per_process, cols)
    
    # Синхронизация перед замером
    comm.Barrier()
    start_time = MPI.Wtime()
    
    # Вычисления на GPU (симуляция)
    device_b = hybrid_mat_vec_mult(device_A, device_x)
    
    # Копирование результата (в реальности: cp.asnumpy(device_b))
    host_b = device_b
    
    comm.Barrier()
    end_time = MPI.Wtime()
    
    elapsed = end_time - start_time
    
    if rank == 0:
        print(f"\nMPI+CuPy execution time: {elapsed:.6f} seconds")
        
        # Верификация
        cpu_b = np.dot(host_A, host_x)
        error = np.max(np.abs(cpu_b - host_b))
        print(f"Max error CPU vs GPU: {error:.2e}")
        print(f"Result (first 3 elements): {host_b[:3]}")
        print("="*70)
    
    return elapsed

if __name__ == "__main__":
    elapsed = main()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\nNote: Simulated CuPy operations through NumPy")
        print("Real CuPy would execute on GPU ~10x faster")
