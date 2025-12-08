#!/usr/bin/env python3
"""
Скрипт для профилирования параллельных программ
Использует cProfile и собственные измерения времени
"""

from mpi4py import MPI
import numpy as np
import time
import json
import sys
import cProfile
import pstats
import io

# Импортируем функции из наших модулей
sys.path.insert(0, '/home/claude/lab13_optimization/src')
from matrix_vector_original import matrix_vector_multiply_original
from matrix_vector_optimized import matrix_vector_multiply_optimized, matrix_vector_multiply_advanced

def profile_function(func, matrix_size, func_name):
    """
    Профилирование функции с использованием cProfile
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Создаем профайлер
    profiler = cProfile.Profile()
    
    # Профилирование
    profiler.enable()
    exec_time, result = func(matrix_size)
    profiler.disable()
    
    # Сохраняем результаты профилирования
    if rank == 0:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Топ-20 функций
        
        profile_output = s.getvalue()
        
        # Сохраняем в файл
        with open(f'/home/claude/lab13_optimization/profiling_results/{func_name}_size{matrix_size}_rank{rank}.txt', 'w') as f:
            f.write(profile_output)
    
    return exec_time

def run_comprehensive_benchmark():
    """
    Комплексное тестирование всех версий программы
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    test_sizes = [500, 1000, 2000, 3000]
    num_runs = 3
    
    results = {
        'original': {},
        'optimized': {},
        'advanced': {}
    }
    
    for matrix_size in test_sizes:
        if rank == 0:
            print(f"\n=== Тестирование размера матрицы: {matrix_size}x{matrix_size} ===")
        
        # Оригинальная версия
        times_original = []
        for run in range(num_runs):
            exec_time = profile_function(matrix_vector_multiply_original, matrix_size, f'original_run{run}')
            times_original.append(exec_time)
            if rank == 0:
                print(f"Original run {run+1}: {exec_time:.6f} s")
        
        # Оптимизированная версия
        times_optimized = []
        for run in range(num_runs):
            exec_time = profile_function(matrix_vector_multiply_optimized, matrix_size, f'optimized_run{run}')
            times_optimized.append(exec_time)
            if rank == 0:
                print(f"Optimized run {run+1}: {exec_time:.6f} s")
        
        # Продвинутая версия
        times_advanced = []
        for run in range(num_runs):
            exec_time = profile_function(matrix_vector_multiply_advanced, matrix_size, f'advanced_run{run}')
            times_advanced.append(exec_time)
            if rank == 0:
                print(f"Advanced run {run+1}: {exec_time:.6f} s")
        
        if rank == 0:
            results['original'][matrix_size] = {
                'times': times_original,
                'mean': np.mean(times_original),
                'std': np.std(times_original)
            }
            results['optimized'][matrix_size] = {
                'times': times_optimized,
                'mean': np.mean(times_optimized),
                'std': np.std(times_optimized)
            }
            results['advanced'][matrix_size] = {
                'times': times_advanced,
                'mean': np.mean(times_advanced),
                'std': np.std(times_advanced)
            }
    
    if rank == 0:
        # Сохранение результатов
        output_file = f'/home/claude/lab13_optimization/profiling_results/benchmark_results_np{size}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\nРезультаты сохранены в {output_file}")
        
        # Вывод сводки
        print("\n=== СВОДКА РЕЗУЛЬТАТОВ ===")
        print(f"{'Размер':<10} {'Original (s)':<15} {'Optimized (s)':<15} {'Advanced (s)':<15} {'Ускорение 1':<15} {'Ускорение 2':<15}")
        print("-" * 85)
        
        for size in test_sizes:
            orig = results['original'][size]['mean']
            opt = results['optimized'][size]['mean']
            adv = results['advanced'][size]['mean']
            speedup1 = orig / opt
            speedup2 = orig / adv
            
            print(f"{size:<10} {orig:<15.6f} {opt:<15.6f} {adv:<15.6f} {speedup1:<15.2f}x {speedup2:<15.2f}x")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 80)
        print("ПРОФИЛИРОВАНИЕ И БЕНЧМАРКИНГ ПАРАЛЛЕЛЬНЫХ ПРОГРАММ")
        print("=" * 80)
        print(f"Количество процессов: {size}")
        print(f"Количество прогонов для каждого теста: 3")
    
    run_comprehensive_benchmark()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("ПРОФИЛИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 80)

if __name__ == "__main__":
    main()
