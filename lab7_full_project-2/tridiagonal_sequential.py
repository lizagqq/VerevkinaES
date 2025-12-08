#!/usr/bin/env python3
"""
Последовательный метод прогонки (алгоритм Томаса)
"""
import numpy as np
import time

def tridiagonal_solve(a, b, c, d):
    """Последовательный метод прогонки"""
    N = len(b)
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    d = np.array(d, dtype=np.float64)
    
    # Прямой ход
    for i in range(1, N):
        m = a[i-1] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        d[i] = d[i] - m * d[i-1]
    
    # Обратный ход
    x = np.zeros(N, dtype=np.float64)
    x[N-1] = d[N-1] / b[N-1]
    for i in range(N-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x

if __name__ == "__main__":
    print("Тестирование последовательного метода прогонки")
    for N in [100, 1000, 10000]:
        data = np.load(f'system_{N}.npz')
        a, b, c, d = data['a'], data['b'], data['c'], data['d']
        x_true = data['x_true']
        
        start = time.time()
        x = tridiagonal_solve(a, b, c, d)
        elapsed = time.time() - start
        
        error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        print(f"N={N:5d}: время={elapsed:.6f}с, ошибка={error:.2e}")
