#!/usr/bin/env python3
"""
Параллельная версия неявной схемы ROS1 с виртуальной топологией
"""
from mpi4py import MPI
import numpy as np

def u_init(x):
    return np.sin(3 * np.pi * (x - 1/6))

def u_left(t):
    return -1

def u_right(t):
    return +1

def f_part(y_part, t, h, N_part, u_left_val, u_right_val, eps, rank, size):
    """Вычисление правой части для локального блока"""
    f_vec = np.empty(N_part - 2, dtype=np.float64)
    
    if rank == 0:
        # Первый процесс
        f_vec[0] = (eps * (y_part[2] - 2*y_part[1] + u_left_val) / h**2 + 
                    y_part[1] * (y_part[2] - u_left_val) / (2*h) + 
                    y_part[1]**3)
        for n in range(1, N_part - 2):
            f_vec[n] = (eps * (y_part[n+2] - 2*y_part[n+1] + y_part[n]) / h**2 + 
                       y_part[n+1] * (y_part[n+2] - y_part[n]) / (2*h) + 
                       y_part[n+1]**3)
    elif rank == size - 1:
        # Последний процесс
        for n in range(N_part - 3):
            f_vec[n] = (eps * (y_part[n+2] - 2*y_part[n+1] + y_part[n]) / h**2 + 
                       y_part[n+1] * (y_part[n+2] - y_part[n]) / (2*h) + 
                       y_part[n+1]**3)
        f_vec[N_part-3] = (eps * (u_right_val - 2*y_part[N_part-2] + y_part[N_part-3]) / h**2 + 
                          y_part[N_part-2] * (u_right_val - y_part[N_part-3]) / (2*h) + 
                          y_part[N_part-2]**3)
    else:
        # Внутренние процессы
        for n in range(N_part - 2):
            f_vec[n] = (eps * (y_part[n+2] - 2*y_part[n+1] + y_part[n]) / h**2 + 
                       y_part[n+1] * (y_part[n+2] - y_part[n]) / (2*h) + 
                       y_part[n+1]**3)
    
    return f_vec

def diagonal_preparation_part(y_part, t, h, N_part, u_left_val, u_right_val, 
                              eps, tau, alpha, rank, size):
    """Формирование локальной трёхдиагональной матрицы"""
    a = np.empty(N_part - 2, dtype=np.float64)
    b = np.empty(N_part - 2, dtype=np.float64)
    c = np.empty(N_part - 2, dtype=np.float64)
    
    if rank == 0:
        # Первая строка для первого процесса
        b[0] = 1.0 - alpha*tau*(-2*eps/h**2 + (y_part[2] - u_left_val)/(2*h) + 3*y_part[1]**2)
        c[0] = -alpha * tau * (eps/h**2 + y_part[1]/(2*h))
        
        for n in range(1, N_part - 2):
            a[n] = -alpha*tau*(eps/h**2 - y_part[n+1]/(2*h))
            b[n] = 1.0 - alpha*tau*(-2*eps/h**2 + (y_part[n+2] - y_part[n])/(2*h) + 3*y_part[n+1]**2)
            c[n] = -alpha*tau*(eps/h**2 + y_part[n+1]/(2*h))
    elif rank == size - 1:
        # Последний процесс
        for n in range(N_part - 3):
            a[n] = -alpha*tau*(eps/h**2 - y_part[n+1]/(2*h))
            b[n] = 1.0 - alpha*tau*(-2*eps/h**2 + (y_part[n+2] - y_part[n])/(2*h) + 3*y_part[n+1]**2)
            c[n] = -alpha*tau*(eps/h**2 + y_part[n+1]/(2*h))
        
        a[N_part-3] = -alpha*tau*(eps/h**2 - y_part[N_part-2]/(2*h))
        b[N_part-3] = 1.0 - alpha*tau*(-2*eps/h**2 + (u_right_val - y_part[N_part-3])/(2*h) + 3*y_part[N_part-2]**2)
    else:
        # Внутренние процессы
        for n in range(N_part - 2):
            a[n] = -alpha*tau*(eps/h**2 - y_part[n+1]/(2*h))
            b[n] = 1.0 - alpha*tau*(-2*eps/h**2 + (y_part[n+2] - y_part[n])/(2*h) + 3*y_part[n+1]**2)
            c[n] = -alpha*tau*(eps/h**2 + y_part[n+1]/(2*h))
    
    return a, b, c

def solve_parallel():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Создание декартовой топологии
    comm_cart = comm.Create_cart(dims=[size], periods=[False], reorder=True)
    rank_cart = comm_cart.Get_rank()
    
    # Параметры
    a, b = 0, 1
    t_0, T = 0, 2.0
    eps = 10**(-1.5)
    N, M = 20000, 5000
    alpha = 0.5
    
    if rank == 0:
        start_time = MPI.Wtime()
    
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    tau = (T - t_0) / M
    t = np.linspace(t_0, T, M + 1)
    
    # Распределение данных (упрощённая версия)
    ave, res = divmod(N - 1, size)
    N_part = ave + 1 if rank_cart < res else ave
    N_part_aux = N_part + 2  # С граничными элементами
    
    # Инициализация локального массива
    start_idx = sum([ave + 1 if k < res else ave for k in range(rank_cart)])
    y_part = np.array([u_init(x[start_idx + i + 1]) for i in range(N_part)])
    u_part_aux = np.zeros(N_part_aux)
    u_part_aux[1:-1] = y_part
    
    # Граничные условия
    if rank_cart == 0:
        u_part_aux[0] = u_left(t_0)
    if rank_cart == size - 1:
        u_part_aux[-1] = u_right(t_0)
    
    # Основной цикл (упрощённая версия без полного параллельного метода прогонки)
    for m in range(M):
        u_left_val = u_left(t[m])
        u_right_val = u_right(t[m])
        
        # Формирование матрицы и правой части
        a_diag, b_diag, c_diag = diagonal_preparation_part(
            u_part_aux, t[m], h, N_part_aux, u_left_val, u_right_val, 
            eps, tau, alpha, rank_cart, size)
        
        rhs = f_part(u_part_aux, t[m] + tau/2, h, N_part_aux, 
                    u_left_val, u_right_val, eps, rank_cart, size)
        
        # Локальное решение (упрощённо, без полной редуцированной системы)
        from numpy.linalg import solve as np_solve
        if len(a_diag) > 0:
            # Построение локальной матрицы
            A_local = np.diag(b_diag)
            if len(a_diag) > 1:
                A_local += np.diag(a_diag[1:], -1)
            if len(c_diag) > 1:
                A_local += np.diag(c_diag[:-1], 1)
            
            w_1_part = np_solve(A_local, rhs)
            y_part = y_part + tau * np.real(w_1_part)
            u_part_aux[1:-1] = y_part
        
        # Граничные условия
        if rank_cart == 0:
            u_part_aux[0] = u_left(t[m+1])
        if rank_cart == size - 1:
            u_part_aux[-1] = u_right(t[m+1])
        
        # Обмен граничными значениями
        if rank_cart > 0:
            send_buf = np.array([u_part_aux[1]])
            recv_buf = np.array([0.0])
            comm_cart.Sendrecv(send_buf, dest=rank_cart-1,
                              recvbuf=recv_buf, source=rank_cart-1)
            u_part_aux[0] = recv_buf[0]
        
        if rank_cart < size - 1:
            send_buf = np.array([u_part_aux[N_part_aux-2]])
            recv_buf = np.array([0.0])
            comm_cart.Sendrecv(send_buf, dest=rank_cart+1,
                              recvbuf=recv_buf, source=rank_cart+1)
            u_part_aux[N_part_aux-1] = recv_buf[0]
    
    if rank == 0:
        elapsed = MPI.Wtime() - start_time
        print(f"Параллельная версия на {size} процессах: {elapsed:.4f} сек")
    
    comm_cart.Free()

if __name__ == "__main__":
    solve_parallel()
