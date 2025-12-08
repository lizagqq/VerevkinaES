#!/usr/bin/env python3
"""
Оптимизированная параллельная версия с Sendrecv
"""
from mpi4py import MPI
import numpy as np

def u_init(x):
    return np.sin(3 * np.pi * (x - 1/6))

def u_left(t):
    return -1

def u_right(t):
    return +1

def solve_parallel_sendrecv():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Создание декартовой топологии
    comm_cart = comm.Create_cart(dims=[size], periods=[False], reorder=True)
    rank_cart = comm_cart.Get_rank()
    
    # Определение соседей
    left_rank, right_rank = comm_cart.Shift(0, 1)
    
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
    N_part = ave + 1 if rank_cart < res else ave
    start_idx = sum([ave + 1 if k < res else ave for k in range(rank_cart)])
    
    # Локальные массивы с граничными элементами
    N_part_aux = N_part + 2
    u_part_aux = np.zeros((M+1, N_part_aux))
    
    # Начальные условия
    for i in range(N_part):
        global_idx = start_idx + i
        u_part_aux[0, i+1] = u_init(x[global_idx])
    
    # Граничные условия
    if rank_cart == 0:
        for m in range(M+1):
            u_part_aux[m, 0] = u_left(t[m])
    if rank_cart == size - 1:
        for m in range(M+1):
            u_part_aux[m, -1] = u_right(t[m])
    
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
        
        # Обмен граничными значениями с соседями
        if rank_cart > 0:
            # Отправить левому, получить от левого
            send_buf = np.array([u_part_aux[m+1, 1]])
            recv_buf = np.array([0.0])
            comm_cart.Sendrecv(send_buf, dest=left_rank, 
                              recvbuf=recv_buf, source=left_rank)
            u_part_aux[m+1, 0] = recv_buf[0]
        
        if rank_cart < size - 1:
            # Отправить правому, получить от правого
            send_buf = np.array([u_part_aux[m+1, N_part_aux-2]])
            recv_buf = np.array([0.0])
            comm_cart.Sendrecv(send_buf, dest=right_rank,
                              recvbuf=recv_buf, source=right_rank)
            u_part_aux[m+1, N_part_aux-1] = recv_buf[0]
    
    if rank == 0:
        elapsed = MPI.Wtime() - start_time
        print(f"Sendrecv на {size} процессах: {elapsed:.4f} сек")
    
    comm_cart.Free()

if __name__ == "__main__":
    solve_parallel_sendrecv()
