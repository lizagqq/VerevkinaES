#!/usr/bin/env python3
"""
Часть 4: Анализ влияния числа потоков
Исследование OMP_NUM_THREADS на производительность
"""
from mpi4py import MPI
import numpy as np
import os
import time
import json

def setup_threading(num_threads):
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

def hybrid_cg(A_part, b_part, N, comm, max_iter=50):
    """Упрощённый метод CG для исследования"""
    x = np.zeros(N, dtype=np.float64)
    p = np.zeros(N, dtype=np.float64)
    r = np.zeros(N, dtype=np.float64)
    q = np.zeros(N, dtype=np.float64)
    
    Ax_local = np.dot(A_part, x)
    r_temp = np.dot(A_part.T, Ax_local - b_part)
    comm.Allreduce([r_temp, MPI.DOUBLE], [r, MPI.DOUBLE], op=MPI.SUM)
    p[:] = r
    
    for _ in range(max_iter):
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
        
        beta = np.dot(r_new, r_new) / r_dot_r
        p = r_new + beta * p
        r[:] = r_new
    
    return x

def thread_scaling_study():
    """Исследование масштабируемости по потокам"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print("ИССЛЕДОВАНИЕ ВЛИЯНИЯ ЧИСЛА ПОТОКОВ")
        print("="*70)
    
    # Параметры
    N = 2000
    local_rows = N // size
    
    # Данные
    np.random.seed(rank * 100)
    A_part = np.random.rand(local_rows, N).astype(np.float64) * 0.1
    for i in range(local_rows):
        global_row = rank * local_rows + i
        if global_row < N:
            A_part[i, global_row] += 5.0
    b_part = np.random.rand(local_rows).astype(np.float64)
    
    # Тестирование различных конфигураций потоков
    thread_configs = [1, 2, 4, 6, 8, 12, 16]
    results = {}
    
    for threads in thread_configs:
        if rank == 0:
            print(f"\nТест с {threads} потоками...")
        
        setup_threading(threads)
        
        comm.Barrier()
        start = MPI.Wtime()
        
        x = hybrid_cg(A_part, b_part, N, comm, max_iter=50)
        
        comm.Barrier()
        elapsed = MPI.Wtime() - start
        
        results[threads] = elapsed
        
        if rank == 0:
            print(f"  Время: {elapsed:.6f} сек")
    
    # Анализ результатов
    if rank == 0:
        # Вычисление ускорения
        baseline = results[1]
        
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЯ")
        print("="*70)
        print(f"{'Потоки':<10} {'Время (сек)':<15} {'Ускорение':<15} {'Эффективность (%)':<20}")
        print("-"*70)
        
        for threads in sorted(results.keys()):
            time_val = results[threads]
            speedup = baseline / time_val
            efficiency = (speedup / threads) * 100
            print(f"{threads:<10} {time_val:<15.6f} {speedup:<15.2f} {efficiency:<20.1f}")
        
        # Сохранение
        with open('thread_scaling.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nРезультаты сохранены в thread_scaling.json")
    
    return results

if __name__ == "__main__":
    thread_scaling_study()
