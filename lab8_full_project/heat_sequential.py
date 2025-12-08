#!/usr/bin/env python3
"""
Последовательная версия явной схемы для уравнения теплопроводности
"""
import numpy as np
import time
import matplotlib.pyplot as plt

def u_init(x):
    """Начальное условие"""
    return np.sin(3 * np.pi * (x - 1/6))

def u_left(t):
    """Граничное условие слева"""
    return -1

def u_right(t):
    """Граничное условие справа"""
    return +1

def solve_sequential(N, M, a=0, b=1, t_0=0, T=6, eps=10**(-1.5)):
    """
    Последовательное решение явной схемой
    
    Параметры:
    N - число точек по пространству
    M - число точек по времени
    """
    # Создание сетки
    x, h = np.linspace(a, b, N+1, retstep=True)
    t, tau = np.linspace(t_0, T, M+1, retstep=True)
    
    # Инициализация решения
    u = np.empty((M+1, N+1))
    
    # Начальное условие
    for n in range(N+1):
        u[0, n] = u_init(x[n])
    
    # Граничные условия
    for m in range(M+1):
        u[m, 0] = u_left(t[m])
        u[m, -1] = u_right(t[m])
    
    # Основной цикл по времени
    start_time = time.time()
    
    for m in range(M):
        for n in range(1, N):
            # Вторая производная
            d2 = (u[m, n+1] - 2*u[m, n] + u[m, n-1]) / h**2
            # Первая производная
            d1 = (u[m, n+1] - u[m, n-1]) / (2*h)
            # Явная схема
            u[m+1, n] = (u[m, n] + 
                        eps * tau * d2 + 
                        tau * u[m, n] * d1 + 
                        tau * u[m, n]**3)
    
    elapsed_time = time.time() - start_time
    
    return u, x, t, elapsed_time

if __name__ == "__main__":
    print("="*70)
    print("ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ЯВНОЙ СХЕМЫ")
    print("="*70)
    
    # Параметры задачи
    test_cases = [
        (200, 20000, "Малая задача"),
        (400, 100000, "Средняя задача"),
        (800, 300000, "Большая задача")
    ]
    
    for N, M, description in test_cases:
        print(f"\n{description}: N={N}, M={M}")
        
        u, x, t, elapsed = solve_sequential(N, M)
        
        print(f"  Время выполнения: {elapsed:.4f} сек")
        print(f"  Размер сетки: {N+1} × {M+1}")
        print(f"  u[0,0] = {u[0,0]:.6f}, u[-1,-1] = {u[-1,-1]:.6f}")
        
        # Сохранение результата для большой задачи
        if N == 800:
            np.savez('solution_sequential.npz', u=u, x=x, t=t, elapsed=elapsed)
            print(f"  Результат сохранён в solution_sequential.npz")
    
    print("\n" + "="*70)
    print("ВЕРИФИКАЦИЯ ЗАВЕРШЕНА")
    print("="*70)
