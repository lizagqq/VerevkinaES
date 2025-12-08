#!/usr/bin/env python3
"""
Часть 1: Базовые асинхронные операции в топологии кольца
"""
from mpi4py import MPI
import numpy as np
import time

def ring_async_basic():
    """Базовый обмен в кольце с Isend/Irecv"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Одиночное значение
    a = np.array([rank], dtype=np.int32)
    b = np.array([100], dtype=np.int32)
    
    requests = [MPI.REQUEST_NULL for _ in range(2)]
    
    if rank == 0:
        requests[0] = comm.Isend([a, 1, MPI.INT], dest=size-1, tag=0)
        requests[1] = comm.Irecv([b, 1, MPI.INT], source=1, tag=0)
    elif rank == size - 1:
        requests[0] = comm.Isend([a, 1, MPI.INT], dest=size-2, tag=0)
        requests[1] = comm.Irecv([b, 1, MPI.INT], source=0, tag=0)
    else:
        requests[0] = comm.Isend([a, 1, MPI.INT], dest=rank-1, tag=0)
        requests[1] = comm.Irecv([b, 1, MPI.INT], source=rank+1, tag=0)
    
    MPI.Request.Waitall(requests)
    
    if rank == 0:
        print(f"Базовый обмен: процесс {rank} получил {b[0]}")

def ring_async_array():
    """Обмен массивами из 10 элементов"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Массив из 10 элементов
    a = np.arange(rank*10, (rank+1)*10, dtype=np.int32)
    b = np.zeros(10, dtype=np.int32)
    
    requests = [MPI.REQUEST_NULL for _ in range(2)]
    
    if rank == 0:
        requests[0] = comm.Isend([a, 10, MPI.INT], dest=size-1, tag=0)
        requests[1] = comm.Irecv([b, 10, MPI.INT], source=1, tag=0)
    elif rank == size - 1:
        requests[0] = comm.Isend([a, 10, MPI.INT], dest=size-2, tag=0)
        requests[1] = comm.Irecv([b, 10, MPI.INT], source=0, tag=0)
    else:
        requests[0] = comm.Isend([a, 10, MPI.INT], dest=rank-1, tag=0)
        requests[1] = comm.Irecv([b, 10, MPI.INT], source=rank+1, tag=0)
    
    MPI.Request.Waitall(requests)
    
    if rank == 0:
        print(f"Обмен массивами: процесс {rank} получил {b[:3]}...")

def ring_async_with_compute():
    """Обмен с вычислениями между Isend/Irecv и Waitall"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 1000
    a = np.random.rand(N)
    b = np.zeros(N)
    
    start = MPI.Wtime()
    
    # Начало асинхронного обмена
    requests = [MPI.REQUEST_NULL for _ in range(2)]
    
    dest = (rank - 1) % size
    source = (rank + 1) % size
    
    requests[0] = comm.Isend([a, N, MPI.DOUBLE], dest=dest, tag=0)
    requests[1] = comm.Irecv([b, N, MPI.DOUBLE], source=source, tag=0)
    
    # Вычисления во время передачи данных
    c = np.zeros(N)
    for _ in range(100):
        c += np.sin(a) * np.cos(a)
    
    # Ожидание завершения коммуникаций
    MPI.Request.Waitall(requests)
    
    elapsed = MPI.Wtime() - start
    
    if rank == 0:
        print(f"Обмен с вычислениями: {elapsed:.6f} сек")
        print(f"  Получено: {b[:3]}, Вычислено: {c[:3]}")
    
    return elapsed

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*70)
        print("ЧАСТЬ 1: БАЗОВЫЕ АСИНХРОННЫЕ ОПЕРАЦИИ")
        print("="*70)
    
    # Задание 1: Базовый обмен
    ring_async_basic()
    comm.Barrier()
    
    # Задание 2: Обмен массивами
    if rank == 0:
        print()
    ring_async_array()
    comm.Barrier()
    
    # Задание 3: Обмен с вычислениями
    if rank == 0:
        print()
    elapsed = ring_async_with_compute()
    
    if rank == 0:
        print("\n" + "="*70)
        print("ЧАСТЬ 1 ЗАВЕРШЕНА")
        print("="*70)
