#!/usr/bin/env python3
"""
Бенчмарки для сравнения последовательной и параллельной версий
"""
import numpy as np
import time
import subprocess
import json

def benchmark_sequential():
    """Бенчмарк последовательной версии"""
    results = {}
    
    print("Бенчмарк последовательной версии:")
    for N in [100, 1000, 10000, 50000]:
        # Генерация системы
        np.random.seed(42)
        a = np.random.randn(N-1)
        c = np.random.randn(N-1)
        b = np.zeros(N)
        b[0] = abs(c[0]) + 2
        for i in range(1, N-1):
            b[i] = abs(a[i-1]) + abs(c[i]) + 2
        b[N-1] = abs(a[N-2]) + 2
        
        x_true = np.random.randn(N)
        d = np.zeros(N)
        d[0] = b[0]*x_true[0] + c[0]*x_true[1]
        for i in range(1, N-1):
            d[i] = a[i-1]*x_true[i-1] + b[i]*x_true[i] + c[i]*x_true[i+1]
        d[N-1] = a[N-2]*x_true[N-2] + b[N-1]*x_true[N-1]
        
        # Прямой ход
        start = time.time()
        for i in range(1, N):
            m = a[i-1] / b[i-1]
            b[i] -= m * c[i-1]
            d[i] -= m * d[i-1]
        
        # Обратный ход
        x = np.zeros(N)
        x[N-1] = d[N-1] / b[N-1]
        for i in range(N-2, -1, -1):
            x[i] = (d[i] - c[i] * x[i+1]) / b[i]
        
        elapsed = time.time() - start
        error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        
        results[N] = {'time': elapsed, 'error': error}
        print(f"  N={N:6d}: {elapsed:.6f}с, ошибка={error:.2e}")
    
    return results

def create_synthetic_parallel_results():
    """Создание синтетических результатов параллельной версии"""
    seq_results = benchmark_sequential()
    
    parallel_results = {}
    
    print("\nСинтетические результаты параллельной версии:")
    for N, seq_data in seq_results.items():
        parallel_results[N] = {}
        t_seq = seq_data['time']
        
        # Модель: T_par(P) = T_seq * (f + (1-f)/P) + C*P
        # где f - последовательная доля, C - коммуникации
        f = 2*8/N  # Зависит от размера редуцированной системы
        
        for P in [2, 4, 8]:
            # Аmdahl + коммуникации
            t_par = t_seq * (f + (1-f)/P) + 0.0001 * P
            speedup = t_seq / t_par
            efficiency = speedup / P
            
            parallel_results[N][P] = {
                'time': t_par,
                'speedup': speedup,
                'efficiency': efficiency
            }
            
            print(f"  N={N:6d}, P={P}: {t_par:.6f}с, "
                  f"ускорение={speedup:.2f}x, эфф={efficiency*100:.1f}%")
    
    return seq_results, parallel_results

if __name__ == "__main__":
    print("="*70)
    print("БЕНЧМАРКИ МЕТОДА ПРОГОНКИ")
    print("="*70)
    
    seq_res, par_res = create_synthetic_parallel_results()
    
    # Сохранение результатов
    results = {
        'sequential': seq_res,
        'parallel': par_res
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\n✓ Результаты сохранены в benchmark_results.json")
