#!/usr/bin/env python3
"""
Параллельная версия с двумерной декомпозицией (по x и y)
"""
from mpi4py import MPI
import numpy as np

def u_init(x, y, eps=10**(-1.5)):
    return 0.5 * np.tanh(1/eps * ((x-0.5)**2 + (y-0.5)**2 - 0.35**2)) - 0.17

def u_left(y, t):
    return 0.33

def u_right(y, t):
    return 0.33

def u_top(x, t):
    return 0.33

def u_bottom(x, t):
    return 0.33

def auxiliary_arrays_determination(M, num):
    ave, res = divmod(M, num)
    rcounts = np.empty(num, dtype=np.int32)
    displs = np.empty(num, dtype=np.int32)
    for k in range(num):
        rcounts[k] = ave + 1 if k < res else ave
        displs[k] = 0 if k == 0 else displs[k-1] + rcounts[k-1]
    return rcounts, displs

def solve_2d_2d_decomposition():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Двумерная топология (сетка)
    num_row = num_col = int(np.sqrt(size))
    if num_row * num_col != size:
        if rank == 0:
            print(f"Число процессов должно быть полным квадратом! size={size}")
        return
    
    comm_cart = comm.Create_cart(dims=(num_row, num_col), 
                                 periods=(False, False), reorder=True)
    rank_cart = comm_cart.Get_rank()
    my_row, my_col = comm_cart.Get_coords(rank_cart)
    
    # Параметры
    a, b, c, d = -2, 2, -2, 2
    t_0, T = 0, 4
    eps = 10**(-1.5)
    N_x, N_y, M = 200, 200, 4000
    
    if rank == 0:
        start_time = MPI.Wtime()
    
    h_x = (b - a) / N_x
    h_y = (d - c) / N_y
    tau = (T - t_0) / M
    
    x = np.linspace(a, b, N_x+1)
    y = np.linspace(c, d, N_y+1)
    t = np.linspace(t_0, T, M+1)
    
    # Определение размеров локального блока
    rcounts_N_x, displs_N_x = auxiliary_arrays_determination(N_x+1, num_col)
    rcounts_N_y, displs_N_y = auxiliary_arrays_determination(N_y+1, num_row)
    
    N_x_part = rcounts_N_x[my_col]
    N_y_part = rcounts_N_y[my_row]
    
    # С граничными элементами
    if my_col in [0, num_col-1]:
        N_x_part_aux = N_x_part + 1
    else:
        N_x_part_aux = N_x_part + 2
    
    if my_row in [0, num_row-1]:
        N_y_part_aux = N_y_part + 1
    else:
        N_y_part_aux = N_y_part + 2
    
    displs_N_x_aux = displs_N_x - 1
    displs_N_x_aux[0] = 0
    displs_N_y_aux = displs_N_y - 1
    displs_N_y_aux[0] = 0
    
    displ_x_aux = displs_N_x_aux[my_col]
    displ_y_aux = displs_N_y_aux[my_row]
    
    # Локальные массивы
    u_part_aux = np.empty((M+1, N_x_part_aux, N_y_part_aux), dtype=np.float64)
    
    # Начальные условия
    for i in range(N_x_part_aux):
        global_i = displ_x_aux + i
        for j in range(N_y_part_aux):
            global_j = displ_y_aux + j
            u_part_aux[0, i, j] = u_init(x[global_i], y[global_j], eps)
    
    # Граничные условия
    for m in range(M+1):
        if my_col == 0:
            for j in range(N_y_part_aux):
                u_part_aux[m, 0, j] = u_left(y[displ_y_aux + j], t[m])
        if my_col == num_col - 1:
            for j in range(N_y_part_aux):
                u_part_aux[m, -1, j] = u_right(y[displ_y_aux + j], t[m])
        if my_row == 0:
            for i in range(N_x_part_aux):
                u_part_aux[m, i, 0] = u_bottom(x[displ_x_aux + i], t[m])
        if my_row == num_row - 1:
            for i in range(N_x_part_aux):
                u_part_aux[m, i, -1] = u_top(x[displ_x_aux + i], t[m])
    
    # Основной цикл
    for m in range(M):
        # Вычисления
        for i in range(1, N_x_part_aux-1):
            for j in range(1, N_y_part_aux-1):
                d2x = (u_part_aux[m, i+1, j] - 2*u_part_aux[m, i, j] + u_part_aux[m, i-1, j]) / h_x**2
                d2y = (u_part_aux[m, i, j+1] - 2*u_part_aux[m, i, j] + u_part_aux[m, i, j-1]) / h_y**2
                d1x = (u_part_aux[m, i+1, j] - u_part_aux[m, i-1, j]) / (2*h_x)
                d1y = (u_part_aux[m, i, j+1] - u_part_aux[m, i, j-1]) / (2*h_y)
                
                u_part_aux[m+1, i, j] = (u_part_aux[m, i, j] + 
                                        tau * (eps * (d2x + d2y) + 
                                              u_part_aux[m, i, j] * (d1x + d1y) + 
                                              u_part_aux[m, i, j]**3))
        
        # Обмен по горизонтали (x)
        if my_col > 0:
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m+1, 1, :], N_y_part_aux, MPI.DOUBLE],
                dest=my_row*num_col + (my_col-1), sendtag=0,
                recvbuf=[u_part_aux[m+1, 0, :], N_y_part_aux, MPI.DOUBLE],
                source=my_row*num_col + (my_col-1), recvtag=MPI.ANY_TAG
            )
        
        if my_col < num_col-1:
            comm_cart.Sendrecv(
                sendbuf=[u_part_aux[m+1, -2, :], N_y_part_aux, MPI.DOUBLE],
                dest=my_row*num_col + (my_col+1), sendtag=0,
                recvbuf=[u_part_aux[m+1, -1, :], N_y_part_aux, MPI.DOUBLE],
                source=my_row*num_col + (my_col+1), recvtag=MPI.ANY_TAG
            )
        
        # Обмен по вертикали (y)
        if my_row > 0:
            temp_send = u_part_aux[m+1, :, 1].copy()
            temp_recv = np.empty(N_x_part_aux, dtype=np.float64)
            comm_cart.Sendrecv(
                sendbuf=[temp_send, N_x_part_aux, MPI.DOUBLE],
                dest=(my_row-1)*num_col + my_col, sendtag=0,
                recvbuf=[temp_recv, N_x_part_aux, MPI.DOUBLE],
                source=(my_row-1)*num_col + my_col, recvtag=MPI.ANY_TAG
            )
            u_part_aux[m+1, :, 0] = temp_recv
        
        if my_row < num_row-1:
            temp_send = u_part_aux[m+1, :, -2].copy()
            temp_recv = np.empty(N_x_part_aux, dtype=np.float64)
            comm_cart.Sendrecv(
                sendbuf=[temp_send, N_x_part_aux, MPI.DOUBLE],
                dest=(my_row+1)*num_col + my_col, sendtag=0,
                recvbuf=[temp_recv, N_x_part_aux, MPI.DOUBLE],
                source=(my_row+1)*num_col + my_col, recvtag=MPI.ANY_TAG
            )
            u_part_aux[m+1, :, -1] = temp_recv
    
    if rank == 0:
        elapsed = MPI.Wtime() - start_time
        print(f"2D декомпозиция на {size} процессах: {elapsed:.4f} сек")
    
    comm_cart.Free()

if __name__ == "__main__":
    solve_2d_2d_decomposition()
