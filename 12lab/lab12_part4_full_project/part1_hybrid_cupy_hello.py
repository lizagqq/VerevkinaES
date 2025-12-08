#!/usr/bin/env python3
"""
Часть 4.1: Базовая гибридная программа MPI + CuPy
Демонстрация работы MPI и CuPy (симуляция GPU через NumPy)
"""
from mpi4py import MPI
import numpy as np
# import cupy as cp  # В реальности используется CuPy
import time

def simulate_gpu_operations(data, rank):
    """
    Симуляция GPU операций через NumPy
    В реальности: cp.arange(), cp.dot() и т.д.
    """
    # Имитация вычислений на GPU
    result = data * 2 + 1
    time.sleep(0.001)  # Имитация времени GPU
    return result

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"MPI process {rank}/{size} initialized")
    
    # Создание данных (в реальности на GPU через CuPy)
    # device_data = cp.arange(10, dtype=cp.float32) + rank * 100
    device_data = np.arange(10, dtype=np.float32) + rank * 100
    
    # Вычисления (в реальности на GPU)
    device_data = simulate_gpu_operations(device_data, rank)
    
    # Копирование на CPU для MPI (в реальности: cp.asnumpy(device_data))
    host_data = device_data
    
    # Сбор данных со всех процессов
    all_data = None
    if rank == 0:
        all_data = np.empty((size, 10), dtype=np.float32)
    
    comm.Gather(host_data, all_data, root=0)
    
    if rank == 0:
        print("\n" + "="*70)
        print("Collected data from all 'GPUs':")
        print("="*70)
        for i in range(size):
            print(f"Process {i}: {all_data[i][:5]}...")
        
        print("\nNote: This is a simulation of MPI+CuPy")
        print("In reality, each process would use CuPy on its GPU")
        print("="*70)

if __name__ == "__main__":
    main()
