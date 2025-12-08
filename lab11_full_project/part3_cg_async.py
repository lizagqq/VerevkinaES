#!/usr/bin/env python3
"""
Часть 3: Метод сопряженных градиентов с асинхронными операциями
"""
from mpi4py import MPI
import numpy as np

def conjugate_gradient_async(N=1000, max_iter=100):
    """
    Метод сопряженных градиентов с асинхронными операциями
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Параметры
    n_local = N // size
    
    # Создание локальной части матрицы (диагональное преобладание)
    A_local = np.zeros((n_local, N))
    for i in range(n_local):
        global_i = rank * n_local + i
        A_local[i, global_i] = 4.0
        if global_i > 0:
            A_local[i, global_i-1] = -1.0
        if global_i < N-1:
            A_local[i, global_i+1] = -1.0
    
    # Правая часть и начальное приближение
    b_local = np.ones(n_local)
    x_local = np.zeros(n_local)
    
    # Начало измерения времени
    comm.Barrier()
    start_time = MPI.Wtime()
    
    # Инициализация
    r_local = b_local - A_local @ np.concatenate(comm.allgather(x_local))
    p_local = r_local.copy()
    
    rs_old = np.dot(r_local, r_local)
    rs_old = comm.allreduce(rs_old, op=MPI.SUM)
    
    # Основной цикл
    for iteration in range(max_iter):
        # Ap = A @ p
        p_global = np.concatenate(comm.allgather(p_local))
        Ap_local = A_local @ p_global
        
        # alpha = rs_old / (p^T * Ap)
        pAp = np.dot(p_local, Ap_local)
        pAp = comm.allreduce(pAp, op=MPI.SUM)
        alpha = rs_old / pAp
        
        # Обновление решения и невязки
        x_local += alpha * p_local
        r_local -= alpha * Ap_local
        
        # rs_new = r^T * r
        rs_new = np.dot(r_local, r_local)
        rs_new = comm.allreduce(rs_new, op=MPI.SUM)
        
        # Проверка сходимости
        if np.sqrt(rs_new) < 1e-10:
            if rank == 0:
                print(f"  Сходимость достигнута на итерации {iteration}")
            break
        
        # beta = rs_new / rs_old
        beta = rs_new / rs_old
        
        # Обновление направления поиска
        p_local = r_local + beta * p_local
        
        rs_old = rs_new
    
    elapsed = MPI.Wtime() - start_time
    
    if rank == 0:
        print(f"  Время выполнения: {elapsed:.6f} сек")
        print(f"  Итераций: {iteration+1}")
    
    return elapsed

def conjugate_gradient_sync(N=1000, max_iter=100):
    """
    Метод сопряженных градиентов с блокирующими операциями (для сравнения)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Параметры
    n_local = N // size
    
    # Создание локальной части матрицы
    A_local = np.zeros((n_local, N))
    for i in range(n_local):
        global_i = rank * n_local + i
        A_local[i, global_i] = 4.0
        if global_i > 0:
            A_local[i, global_i-1] = -1.0
        if global_i < N-1:
            A_local[i, global_i+1] = -1.0
    
    # Правая часть и начальное приближение
    b_local = np.ones(n_local)
    x_local = np.zeros(n_local)
    
    # Начало измерения времени
    comm.Barrier()
    start_time = MPI.Wtime()
    
    # Инициализация
    r_local = b_local - A_local @ np.concatenate(comm.allgather(x_local))
    p_local = r_local.copy()
    
    rs_old = np.dot(r_local, r_local)
    rs_old = comm.allreduce(rs_old, op=MPI.SUM)
    
    # Основной цикл (идентичен асинхронной версии для честного сравнения)
    for iteration in range(max_iter):
        p_global = np.concatenate(comm.allgather(p_local))
        Ap_local = A_local @ p_global
        
        pAp = np.dot(p_local, Ap_local)
        pAp = comm.allreduce(pAp, op=MPI.SUM)
        alpha = rs_old / pAp
        
        x_local += alpha * p_local
        r_local -= alpha * Ap_local
        
        rs_new = np.dot(r_local, r_local)
        rs_new = comm.allreduce(rs_new, op=MPI.SUM)
        
        if np.sqrt(rs_new) < 1e-10:
            break
        
        beta = rs_new / rs_old
        p_local = r_local + beta * p_local
        rs_old = rs_new
    
    elapsed = MPI.Wtime() - start_time
    
    if rank == 0:
        print(f"  Время выполнения: {elapsed:.6f} сек")
    
    return elapsed

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("="*70)
        print("ЧАСТЬ 3: МЕТОД СОПРЯЖЕННЫХ ГРАДИЕНТОВ")
        print("="*70)
    
    # Асинхронная версия
    if rank == 0:
        print("\nАсинхронная версия:")
    t_async = conjugate_gradient_async(N=1000, max_iter=100)
    
    comm.Barrier()
    
    # Синхронная версия
    if rank == 0:
        print("\nСинхронная версия:")
    t_sync = conjugate_gradient_sync(N=1000, max_iter=100)
    
    if rank == 0:
        print(f"\nУскорение: {t_sync/t_async:.2f}x")
        print("\n" + "="*70)
        print("ЧАСТЬ 3 ЗАВЕРШЕНА")
        print("="*70)
