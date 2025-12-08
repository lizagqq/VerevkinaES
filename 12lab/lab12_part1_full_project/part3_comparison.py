#!/usr/bin/env python3
"""
Часть 3: Сравнительный анализ эффективности
Чистый MPI vs Гибридный подход
"""
from mpi4py import MPI
import numpy as np
import os
import time
import json

def setup_threading(num_threads):
    """Настройка числа потоков"""
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

def pure_mpi_cg(A_part, b_part, N, comm, max_iter=100):
    """Чистый MPI (1 поток на процесс)"""
    setup_threading(1)
    
    rank = comm.Get_rank()
    
    x = np.zeros(N, dtype=np.float64)
    p = np.zeros(N, dtype=np.float64)
    r = np.zeros(N, dtype=np.float64)
    q = np.zeros(N, dtype=np.float64)
    
    # Первая итерация
    Ax_local = np.dot(A_part, x)
    r_temp = np.dot(A_part.T, Ax_local - b_part)
    comm.Allreduce([r_temp, MPI.DOUBLE], [r, MPI.DOUBLE], op=MPI.SUM)
    p[:] = r
    
    # Основной цикл
    for iteration in range(max_iter):
        Ap_local = np.dot(A_part, p)
        q_temp = np.dot(A_part.T, Ap_local)
        comm.Allreduce([q_temp, MPI.DOUBLE], [q, MPI.DOUBLE], op=MPI.SUM)
        
        r_dot_r = np.dot(r, r)
        p_dot_q = np.dot(p, q)
        
        if abs(p_dot_q) < 1e-15:
            break
        
        alpha = r_dot_r / p_dot_q
        x -= alpha * p
        r_new = r - alpha * q
        
        if np.linalg.norm(r_new) < 1e-8:
            break
        
        beta = np.dot(r_new, r_new) / r_dot_r
        p = r_new + beta * p
        r[:] = r_new
    
    return x

def hybrid_cg(A_part, b_part, N, comm, threads, max_iter=100):
    """Гибридный метод (MPI + OpenMP)"""
    setup_threading(threads)
    
    rank = comm.Get_rank()
    
    x = np.zeros(N, dtype=np.float64)
    p = np.zeros(N, dtype=np.float64)
    r = np.zeros(N, dtype=np.float64)
    q = np.zeros(N, dtype=np.float64)
    
    # Первая итерация
    Ax_local = np.dot(A_part, x)
    r_temp = np.dot(A_part.T, Ax_local - b_part)
    comm.Allreduce([r_temp, MPI.DOUBLE], [r, MPI.DOUBLE], op=MPI.SUM)
    p[:] = r
    
    # Основной цикл
    for iteration in range(max_iter):
        Ap_local = np.dot(A_part, p)
        q_temp = np.dot(A_part.T, Ap_local)
        comm.Allreduce([q_temp, MPI.DOUBLE], [q, MPI.DOUBLE], op=MPI.SUM)
        
        r_dot_r = np.dot(r, r)
        p_dot_q = np.dot(p, q)
        
        if abs(p_dot_q) < 1e-15:
            break
        
        alpha = r_dot_r / p_dot_q
        x -= alpha * p
        r_new = r - alpha * q
        
        if np.linalg.norm(r_new) < 1e-8:
            break
        
        beta = np.dot(r_new, r_new) / r_dot_r
        p = r_new + beta * p
        r[:] = r_new
    
    return x

def benchmark():
    """Сравнительное тестирование"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ: MPI vs HYBRID")
        print("="*70)
    
    # Параметры задачи
    N = 2000
    local_rows = N // size
    
    # Генерация данных
    np.random.seed(rank * 100)
    A_part = np.random.rand(local_rows, N).astype(np.float64) * 0.1
    for i in range(local_rows):
        global_row = rank * local_rows + i
        if global_row < N:
            A_part[i, global_row] += 5.0
    b_part = np.random.rand(local_rows).astype(np.float64)
    
    results = {}
    
    # Тест 1: Чистый MPI
    if rank == 0:
        print(f"\n--- Чистый MPI ({size} процессов x 1 поток) ---")
    
    comm.Barrier()
    start = MPI.Wtime()
    x_mpi = pure_mpi_cg(A_part, b_part, N, comm)
    comm.Barrier()
    time_mpi = MPI.Wtime() - start
    
    if rank == 0:
        print(f"Время: {time_mpi:.6f} сек")
    
    results['pure_mpi'] = {'time': time_mpi, 'threads': 1}
    
    # Тест 2: Гибридный (разное число потоков)
    thread_configs = [2, 4, 8]
    
    for threads in thread_configs:
        if rank == 0:
            print(f"\n--- Гибридный ({size} процессов x {threads} потоков) ---")
        
        comm.Barrier()
        start = MPI.Wtime()
        x_hybrid = hybrid_cg(A_part, b_part, N, comm, threads)
        comm.Barrier()
        time_hybrid = MPI.Wtime() - start
        
        if rank == 0:
            print(f"Время: {time_hybrid:.6f} сек")
            speedup = time_mpi / time_hybrid
            print(f"Ускорение относительно чистого MPI: {speedup:.2f}x")
        
        results[f'hybrid_{threads}'] = {'time': time_hybrid, 'threads': threads}
    
    # Сохранение результатов
    if rank == 0:
        with open('comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print("="*70)
        print(f"{'Конфигурация':<20} {'Время (сек)':<15} {'Ускорение':<10}")
        print("-"*70)
        
        baseline = results['pure_mpi']['time']
        for config, data in results.items():
            speedup = baseline / data['time']
            print(f"{config:<20} {data['time']:<15.6f} {speedup:<10.2f}")
    
    return results

if __name__ == "__main__":
    benchmark()
