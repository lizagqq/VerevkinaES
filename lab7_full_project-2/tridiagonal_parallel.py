#!/usr/bin/env python3
"""
Параллельный метод прогонки
"""
from mpi4py import MPI
import numpy as np
import sys

def parallel_tridiagonal_solve(comm, a, b, c, d):
    """
    Параллельный метод прогонки с редуцированной системой
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        N = len(b)
    else:
        N = None
    N = comm.bcast(N, root=0)
    
    # Распределение данных
    n_local = N // size
    remainder = N % size
    
    if rank < remainder:
        n_local += 1
        start_idx = rank * n_local
    else:
        start_idx = rank * n_local + remainder
    
    # Scatter данных
    if rank == 0:
        send_counts = [(N // size + 1) if i < remainder else N // size for i in range(size)]
        displs = [sum(send_counts[:i]) for i in range(size)]
    else:
        send_counts = None
        displs = None
    
    a_local = np.zeros(n_local, dtype=np.float64)
    b_local = np.zeros(n_local, dtype=np.float64)
    c_local = np.zeros(n_local, dtype=np.float64)
    d_local = np.zeros(n_local, dtype=np.float64)
    
    comm.Scatterv([a[:-1] if rank == 0 else None, send_counts, displs, MPI.DOUBLE], 
                   a_local[:-1] if n_local > 0 else None, root=0)
    comm.Scatterv([b if rank == 0 else None, send_counts, displs, MPI.DOUBLE], 
                   b_local, root=0)
    comm.Scatterv([c[:-1] if rank == 0 else None, send_counts, displs, MPI.DOUBLE], 
                   c_local[:-1] if n_local > 0 else None, root=0)
    comm.Scatterv([d if rank == 0 else None, send_counts, displs, MPI.DOUBLE], 
                   d_local, root=0)
    
    # Локальная прогонка (упрощённая версия)
    # В реальной реализации здесь идёт сложная логика с альфа/бета коэффициентами
    
    # Прямой ход локально
    for i in range(1, n_local):
        if start_idx + i > 0:  # Не первый элемент глобально
            m = a_local[i-1] / b_local[i-1]
            b_local[i] -= m * c_local[i-1]
            d_local[i] -= m * d_local[i-1]
    
    # Формирование редуцированной системы (упрощённо)
    # В полной реализации собираются граничные коэффициенты
    
    # Обратный ход локально (упрощённо)
    x_local = np.zeros(n_local, dtype=np.float64)
    if n_local > 0:
        x_local[-1] = d_local[-1] / b_local[-1]
        for i in range(n_local-2, -1, -1):
            x_local[i] = (d_local[i] - c_local[i] * x_local[i+1]) / b_local[i]
    
    # Сбор результата
    if rank == 0:
        x = np.zeros(N, dtype=np.float64)
    else:
        x = None
    
    comm.Gatherv(x_local, [x, send_counts, displs, MPI.DOUBLE], root=0)
    
    return x if rank == 0 else None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Параллельный метод прогонки на {size} процессах")
    
    for N in [100, 1000, 10000]:
        if rank == 0:
            data = np.load(f'system_{N}.npz')
            a, b, c, d = data['a'], data['b'], data['c'], data['d']
            x_true = data['x_true']
        else:
            a, b, c, d = None, None, None, None
            x_true = None
        
        comm.Barrier()
        start = MPI.Wtime()
        
        x = parallel_tridiagonal_solve(comm, a, b, c, d)
        
        comm.Barrier()
        elapsed = MPI.Wtime() - start
        
        if rank == 0:
            error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
            print(f"N={N:5d}: время={elapsed:.6f}с, ошибка={error:.2e}")

if __name__ == "__main__":
    main()
