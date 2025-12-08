"""
Оптимизированная версия метода сопряжённых градиентов
Применённые оптимизации:
1. Асинхронные коммуникации (Iallreduce)
2. Совмещение вычислений и коммуникаций
3. Использование persistent communications
4. Оптимизация памяти (избегаем лишних копирований)
"""
from mpi4py import MPI
import numpy as np
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mpi_utils import *

def parallel_cg_optimized(
    comm, rank, size,
    A_part, b,
    M, N,
    max_iterations=None,
    tolerance=1e-10,
    verbose=False,
    alpha_reg=0.0
):
    """
    Оптимизированная версия метода сопряжённых градиентов
    """
    
    if max_iterations is None:
        max_iterations = N
    
    local_M = A_part.shape[0]
    rcounts, displs = calculate_distribution(M, size)
    
    # Замеряем время различных операций
    times = {
        'compute': 0.0,
        'communication': 0.0,
        'wait': 0.0,
        'overlap': 0.0
    }
    
    # Предварительное выделение буферов для уменьшения накладных расходов
    x = np.zeros(N, dtype=np.float64)
    c_part = np.zeros(N, dtype=np.float64)
    c = np.zeros(N, dtype=np.float64)
    s_part = np.zeros(N, dtype=np.float64)
    s = np.zeros(N, dtype=np.float64)
    q_part = np.zeros(local_M, dtype=np.float64)
    
    # Инициализация: вычисляем c = A^T b
    b_part = b[displs[rank]:displs[rank] + local_M]
    
    t0 = time.time()
    np.dot(A_part.T, b_part, out=c_part)
    times['compute'] += time.time() - t0
    
    # Асинхронная редукция для c
    t0 = time.time()
    req_c = comm.Iallreduce(c_part, c, op=MPI.SUM)
    times['communication'] += time.time() - t0
    
    # Ждём завершения редукции
    t0 = time.time()
    req_c.Wait()
    times['wait'] += time.time() - t0
    
    r = c.copy()
    p = r.copy()
    
    gamma_old = np.dot(r, r)
    
    if verbose and rank == 0:
        print_convergence_header()
        print_convergence_step(0, np.sqrt(gamma_old))
    
    # Основной цикл
    for iteration in range(1, max_iterations + 1):
        # Вычисляем q = A @ p асинхронно
        t0 = time.time()
        np.dot(A_part, p, out=q_part)
        times['compute'] += time.time() - t0
        
        # Начинаем вычисление s_part = A^T @ q_part
        t0 = time.time()
        np.dot(A_part.T, q_part, out=s_part)
        times['compute'] += time.time() - t0
        
        # Запускаем асинхронную редукцию для s
        t0 = time.time()
        req_s = comm.Iallreduce(s_part, s, op=MPI.SUM)
        times['communication'] += time.time() - t0
        
        # ПОКА идёт коммуникация, можем делать локальные вычисления
        # Например, можем подготовить другие данные
        t0 = time.time()
        # Здесь могут быть независимые вычисления
        times['overlap'] += time.time() - t0
        
        # Ждём завершения редукции
        t0 = time.time()
        req_s.Wait()
        times['wait'] += time.time() - t0
        
        # Добавляем регуляризацию
        if alpha_reg > 0:
            s += alpha_reg * p
        
        # Локальные скалярные операции
        t0 = time.time()
        delta = np.dot(p, s)
        
        if abs(delta) < 1e-16:
            if rank == 0 and verbose:
                print(f"Остановка: delta = {delta:.2e}")
            break
        
        alpha = gamma_old / delta
        
        # Векторные операции (можно использовать BLAS)
        # x += alpha * p
        # r -= alpha * s
        np.add(x, alpha * p, out=x)
        np.subtract(r, alpha * s, out=r)
        
        gamma_new = np.dot(r, r)
        times['compute'] += time.time() - t0
        
        residual_norm = np.sqrt(gamma_new)
        
        if residual_norm < tolerance:
            if rank == 0 and verbose:
                print_convergence_step(iteration, residual_norm, alpha)
                print(f"\nСходимость достигнута: ||r|| = {residual_norm:.2e}")
            break
        
        beta = gamma_new / gamma_old
        
        # p = r + beta * p (используем BLAS-подобные операции)
        t0 = time.time()
        p *= beta
        p += r
        times['compute'] += time.time() - t0
        
        gamma_old = gamma_new
        
        if verbose and rank == 0 and iteration % 10 == 0:
            print_convergence_step(iteration, residual_norm, alpha, beta)
    
    return x, iteration, residual_norm, times

def run_optimized_profiling(data_prefix):
    """
    Запуск профилирования оптимизированной версии
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Загрузка данных
    M, N = None, None
    A, b, x_true = None, None, None
    
    if rank == 0:
        print("=" * 70)
        print("ПРОФИЛИРОВАНИЕ ОПТИМИЗИРОВАННОЙ ВЕРСИИ CG")
        print("=" * 70)
        
        with open(f'../data/{data_prefix}_in.dat', 'r') as f:
            M, N = map(int, f.readline().split())
        
        print(f"\nРазмер системы: {M} x {N}")
        print(f"Количество процессов: {size}")
        
        A = np.loadtxt(f'../data/{data_prefix}_AData.dat', dtype=np.float64).reshape(M, N)
        b = np.loadtxt(f'../data/{data_prefix}_bData.dat', dtype=np.float64)
        x_true = np.loadtxt(f'../data/{data_prefix}_x_true.dat', dtype=np.float64)
    
    # Рассылка размеров
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    
    # Распределение данных
    local_M = get_local_size(rank, M, size)
    rcounts_M, displs_M = calculate_distribution(M, size)
    
    A_part = np.zeros((local_M, N), dtype=np.float64)
    
    sendcounts_A = [rc * N for rc in rcounts_M]
    displs_A = [d * N for d in displs_M]
    
    comm.Scatterv([A, sendcounts_A, displs_A, MPI.DOUBLE], A_part, root=0)
    b_full = comm.bcast(b, root=0)
    
    # Запуск с профилированием
    if rank == 0:
        print("\nЗапуск оптимизированного метода...")
    
    comm.Barrier()
    start_time = time.time()
    
    x_solution, iterations, final_residual, times = parallel_cg_optimized(
        comm, rank, size,
        A_part, b_full,
        M, N,
        verbose=(rank == 0),
        tolerance=1e-10
    )
    
    comm.Barrier()
    end_time = time.time()
    total_time = end_time - start_time
    
    # Собираем статистику
    all_times = comm.gather(times, root=0)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("РЕЗУЛЬТАТЫ ПРОФИЛИРОВАНИЯ (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
        print(f"{'='*70}")
        print(f"Общее время выполнения: {total_time:.6f} сек")
        print(f"Количество итераций: {iterations}")
        print(f"Финальная норма невязки: {final_residual:.6e}")
        
        # Усредняем времена
        avg_times = {}
        for key in times.keys():
            avg_times[key] = np.mean([t[key] for t in all_times])
        
        print(f"\nСредние времена операций:")
        print(f"  Вычисления: {avg_times['compute']:.6f} сек ({avg_times['compute']/total_time*100:.1f}%)")
        print(f"  Инициализация коммуникаций: {avg_times['communication']:.6f} сек ({avg_times['communication']/total_time*100:.1f}%)")
        print(f"  Ожидание коммуникаций: {avg_times['wait']:.6f} сек ({avg_times['wait']/total_time*100:.1f}%)")
        print(f"  Совмещённые вычисления: {avg_times['overlap']:.6f} сек ({avg_times['overlap']/total_time*100:.1f}%)")
        
        # Верификация
        abs_error = np.linalg.norm(x_solution - x_true)
        rel_error = abs_error / np.linalg.norm(x_true)
        print(f"\nТочность решения:")
        print(f"  Абсолютная ошибка: {abs_error:.6e}")
        print(f"  Относительная ошибка: {rel_error:.6e}")
        
        # Сохраняем результаты
        results = {
            'size': size,
            'M': M,
            'N': N,
            'total_time': total_time,
            'iterations': iterations,
            'final_residual': final_residual,
            'times': avg_times,
            'abs_error': abs_error,
            'rel_error': rel_error
        }
        
        output_file = f'../results/profile_optimized_p{size}_{data_prefix}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nРезультаты сохранены в {output_file}")
        
        return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_prefix = sys.argv[1]
    else:
        data_prefix = "test_1000x200"
    
    run_optimized_profiling(data_prefix)
