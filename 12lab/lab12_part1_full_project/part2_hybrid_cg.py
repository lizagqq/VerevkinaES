#!/usr/bin/env python3
"""
Часть 2: Гибридная реализация метода сопряжённых градиентов
MPI + многопоточный NumPy (OpenMP)
"""
from mpi4py import MPI
import numpy as np
import os
import time

def setup_threading(num_threads):
    """Настройка числа потоков для NumPy/OpenMP"""
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

def conjugate_gradient_hybrid(A_part, b_part, N, comm, max_iter=100, tolerance=1e-8):
    """
    Гибридный метод сопряженных градиентов
    A_part: локальная часть матрицы
    b_part: локальная часть правой части
    N: размер глобальной системы
    """
    rank = comm.Get_rank()
    
    # Инициализация
    x = np.zeros(N, dtype=np.float64)
    p = np.zeros(N, dtype=np.float64)
    r = np.zeros(N, dtype=np.float64)
    q = np.zeros(N, dtype=np.float64)
    
    # Первая итерация: r = A^T * (A*x - b)
    # A*x (многопоточное через NumPy)
    Ax_local = np.dot(A_part, x)
    
    # A^T * (A*x - b)
    r_temp = np.dot(A_part.T, Ax_local - b_part)
    
    # Глобальная редукция
    comm.Allreduce([r_temp, MPI.DOUBLE], [r, MPI.DOUBLE], op=MPI.SUM)
    
    # p = r
    p[:] = r
    
    # Основной цикл
    for iteration in range(max_iter):
        # q = A^T * A * p (многопоточное)
        Ap_local = np.dot(A_part, p)
        q_temp = np.dot(A_part.T, Ap_local)
        comm.Allreduce([q_temp, MPI.DOUBLE], [q, MPI.DOUBLE], op=MPI.SUM)
        
        # alpha = (r^T * r) / (p^T * q)
        r_dot_r = np.dot(r, r)
        p_dot_q = np.dot(p, q)
        
        if abs(p_dot_q) < 1e-15:
            if rank == 0:
                print(f"Деление на ноль на итерации {iteration}")
            break
        
        alpha = r_dot_r / p_dot_q
        
        # x = x - alpha * p
        x -= alpha * p
        
        # r_new = r - alpha * q
        r_new = r - alpha * q
        
        # Проверка сходимости
        r_new_norm = np.linalg.norm(r_new)
        if r_new_norm < tolerance:
            if rank == 0:
                print(f"Сходимость достигнута на итерации {iteration+1}")
            break
        
        # beta = (r_new^T * r_new) / (r^T * r)
        r_new_dot_r_new = np.dot(r_new, r_new)
        beta = r_new_dot_r_new / r_dot_r
        
        # p = r_new + beta * p
        p = r_new + beta * p
        r[:] = r_new
    
    return x

def main():
    """Основная программа"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Настройка потоков (по умолчанию)
    threads = 4
    setup_threading(threads)
    
    if rank == 0:
        print("="*70)
        print("ГИБРИДНЫЙ МЕТОД СОПРЯЖЕННЫХ ГРАДИЕНТОВ (MPI + OpenMP)")
        print("="*70)
        print(f"Конфигурация: {size} процессов x {threads} потоков")
    
    # Параметры задачи
    N = 1000
    local_rows = N // size
    
    if rank == 0:
        print(f"Размер системы: {N} x {N}")
        print(f"Локальных строк на процесс: {local_rows}")
    
    # Генерация локальной части матрицы (диагональное преобладание)
    np.random.seed(rank * 100)
    A_part = np.random.rand(local_rows, N).astype(np.float64) * 0.1
    
    # Усиление диагонали для устойчивости
    for i in range(local_rows):
        global_row = rank * local_rows + i
        if global_row < N:
            A_part[i, global_row] += 5.0
    
    # Генерация правой части
    b_part = np.random.rand(local_rows).astype(np.float64)
    
    # Решение системы
    if rank == 0:
        print("\nРешение системы...")
    
    comm.Barrier()
    start_time = MPI.Wtime()
    
    x_solution = conjugate_gradient_hybrid(A_part, b_part, N, comm, max_iter=100)
    
    comm.Barrier()
    end_time = MPI.Wtime()
    
    elapsed = end_time - start_time
    
    if rank == 0:
        print(f"\nВремя выполнения: {elapsed:.6f} сек")
        
        # Проверка решения
        Ax_local = np.dot(A_part, x_solution)
        residual_local = np.linalg.norm(Ax_local - b_part)
        print(f"Локальная невязка: {residual_local:.4e}")
        print(f"Решение (первые 5 элементов): {x_solution[:5]}")
        
        print("\n" + "="*70)
        print("ЗАВЕРШЕНО")
        print("="*70)
    
    return elapsed

if __name__ == "__main__":
    main()
