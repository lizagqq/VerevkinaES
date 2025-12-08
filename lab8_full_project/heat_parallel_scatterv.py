#!/usr/bin/env python3
"""
Параллельная версия с Scatterv/Gatherv для явной схемы
"""
from mpi4py import MPI
import numpy as np

def u_init(x):
    return np.sin(3 * np.pi * (x - 1/6))

def u_left(t):
    return -1

def u_right(t):
    return +1

def solve_parallel_scatterv():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Параметры задачи
    a, b = (0, 1)
    t_0, T = (0, 6)
    eps = 10**(-1.5)
    N, M = (800, 300000)
    
    if rank == 0:
        start_time = MPI.Wtime()
        x, h = np.linspace(a, b, N+1, retstep=True)
        t, tau = np.linspace(t_0, T, M+1, retstep=True)
        
        # Распределение данных
        ave, res = divmod(N+1, size)
        rcounts = np.array([ave + 1 if k < res else ave for k in range(size)], dtype=np.int32)
        displs = np.array([sum(rcounts[:k]) for k in range(size)], dtype=np.int32)
    else:
        x, h, t, tau = None, None, None, None
        rcounts, displs = None, None
    
    # Рассылка параметров
    h = comm.bcast(h, root=0)
    tau = comm.bcast(tau, root=0)
    
    # Получение размера локального блока
    N_part = np.array(0, dtype=np.int32)
    comm.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)
    
    # Локальные массивы с граничными элементами
    N_part_aux = N_part + 2
    u_part_aux = np.zeros((M+1, N_part_aux))
    
    # Scatter начальных условий
    if rank == 0:
        u_init_global = np.array([u_init(x[i]) for i in range(N+1)])
    else:
        u_init_global = None
    
    u_part = np.zeros(N_part)
    comm.Scatterv([u_init_global, rcounts, displs, MPI.DOUBLE], u_part, root=0)
    u_part_aux[0, 1:-1] = u_part
    
    # Граничные условия
    if rank == 0:
        u_part_aux[:, 0] = u_left(0)
    if rank == size - 1:
        u_part_aux[:, -1] = u_right(0)
    
    # Основной цикл
    for m in range(M):
        # Вычисления
        for n in range(1, N_part_aux-1):
            d2 = (u_part_aux[m, n+1] - 2*u_part_aux[m, n] + u_part_aux[m, n-1]) / h**2
            d1 = (u_part_aux[m, n+1] - u_part_aux[m, n-1]) / (2*h)
            u_part_aux[m+1, n] = (u_part_aux[m, n] + 
                                 eps * tau * d2 + 
                                 tau * u_part_aux[m, n] * d1 + 
                                 tau * u_part_aux[m, n]**3)
        
        # Обмен граничными значениями (упрощённо через Gatherv/Scatterv)
        u_part = u_part_aux[m+1, 1:-1]
        if rank == 0:
            u_global = np.zeros(N+1)
        else:
            u_global = None
        
        comm.Gatherv(u_part, [u_global, rcounts, displs, MPI.DOUBLE], root=0)
        
        if rank == 0 and m < M - 1:
            # Применение граничных условий
            u_global[0] = u_left(t[m+1])
            u_global[-1] = u_right(t[m+1])
        
        comm.Scatterv([u_global, rcounts, displs, MPI.DOUBLE], u_part, root=0)
        u_part_aux[m+1, 1:-1] = u_part
    
    if rank == 0:
        elapsed = MPI.Wtime() - start_time
        print(f"Scatterv/Gatherv на {size} процессах: {elapsed:.4f} сек")

if __name__ == "__main__":
    solve_parallel_scatterv()
