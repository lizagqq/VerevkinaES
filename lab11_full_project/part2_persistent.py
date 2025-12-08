#!/usr/bin/env python3
"""
Часть 2: Отложенные запросы на взаимодействие
"""
from mpi4py import MPI
import numpy as np
import time

def persistent_requests_basic():
    """Многократный обмен с отложенными запросами"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    a = np.array([rank], dtype=np.int32)
    a_recv = np.empty(1, dtype=np.int32)
    
    # Инициализация отложенных запросов
    requests = [MPI.REQUEST_NULL for _ in range(2)]
    requests[0] = comm.Send_init([a, 1, MPI.INT], dest=(rank+1)%size, tag=0)
    requests[1] = comm.Recv_init([a_recv, 1, MPI.INT], source=(rank-1)%size, tag=0)
    
    # 10 итераций обмена
    for iteration in range(10):
        MPI.Prequest.Startall(requests)
        MPI.Request.Waitall(requests)
        a[0] = a_recv[0]
    
    # Освобождение запросов
    for req in requests:
        req.Free()
    
    if rank == 0:
        print(f"Отложенные запросы: процесс {rank} финальное значение {a[0]}")

def persistent_requests_2d():
    """Обмен с двумерными массивами"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 10
    a = np.ones((N, N), dtype=np.float64) * rank
    a_recv = np.empty((N, N), dtype=np.float64)
    
    # Инициализация отложенных запросов
    requests = [MPI.REQUEST_NULL for _ in range(2)]
    requests[0] = comm.Send_init([a, N*N, MPI.DOUBLE], dest=(rank+1)%size, tag=0)
    requests[1] = comm.Recv_init([a_recv, N*N, MPI.DOUBLE], source=(rank-1)%size, tag=0)
    
    # 10 итераций
    for _ in range(10):
        MPI.Prequest.Startall(requests)
        MPI.Request.Waitall(requests)
        a[:] = a_recv
    
    # Освобождение
    for req in requests:
        req.Free()
    
    if rank == 0:
        print(f"2D массивы: процесс {rank} финальное значение {a[0,0]}")

def compare_persistent_vs_sendrecv():
    """Сравнение производительности: отложенные vs Sendrecv_replace"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    N = 1000
    iterations = 100
    
    # Версия 1: Отложенные запросы
    a = np.random.rand(N)
    a_recv = np.empty(N)
    
    requests = [MPI.REQUEST_NULL for _ in range(2)]
    requests[0] = comm.Send_init([a, N, MPI.DOUBLE], dest=(rank+1)%size, tag=0)
    requests[1] = comm.Recv_init([a_recv, N, MPI.DOUBLE], source=(rank-1)%size, tag=0)
    
    comm.Barrier()
    start1 = MPI.Wtime()
    
    for _ in range(iterations):
        MPI.Prequest.Startall(requests)
        MPI.Request.Waitall(requests)
        a[:] = a_recv
    
    elapsed1 = MPI.Wtime() - start1
    
    for req in requests:
        req.Free()
    
    # Версия 2: Sendrecv_replace
    b = np.random.rand(N)
    
    comm.Barrier()
    start2 = MPI.Wtime()
    
    for _ in range(iterations):
        comm.Sendrecv_replace([b, N, MPI.DOUBLE], 
                              dest=(rank+1)%size, sendtag=0,
                              source=(rank-1)%size, recvtag=0)
    
    elapsed2 = MPI.Wtime() - start2
    
    if rank == 0:
        print(f"\nСравнение производительности ({iterations} итераций):")
        print(f"  Отложенные запросы: {elapsed1:.6f} сек")
        print(f"  Sendrecv_replace:   {elapsed2:.6f} сек")
        print(f"  Ускорение: {elapsed2/elapsed1:.2f}x")
    
    return elapsed1, elapsed2

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*70)
        print("ЧАСТЬ 2: ОТЛОЖЕННЫЕ ЗАПРОСЫ")
        print("="*70)
    
    # Задание 1: Базовые отложенные запросы
    persistent_requests_basic()
    comm.Barrier()
    
    # Задание 2: 2D массивы
    if rank == 0:
        print()
    persistent_requests_2d()
    comm.Barrier()
    
    # Задание 3: Сравнение производительности
    t1, t2 = compare_persistent_vs_sendrecv()
    
    if rank == 0:
        print("\n" + "="*70)
        print("ЧАСТЬ 2 ЗАВЕРШЕНА")
        print("="*70)
