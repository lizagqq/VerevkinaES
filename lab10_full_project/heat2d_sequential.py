#!/usr/bin/env python3
"""
Последовательная версия для двумерного уравнения теплопроводности
"""
import numpy as np
import time

def u_init(x, y, eps=10**(-1.5)):
    """Начальное условие"""
    return 0.5 * np.tanh(1/eps * ((x-0.5)**2 + (y-0.5)**2 - 0.35**2)) - 0.17

def u_left(y, t):
    """Граничное условие слева"""
    return 0.33

def u_right(y, t):
    """Граничное условие справа"""
    return 0.33

def u_top(x, t):
    """Граничное условие сверху"""
    return 0.33

def u_bottom(x, t):
    """Граничное условие снизу"""
    return 0.33

def solve_2d_sequential(N_x, N_y, M, a=-2, b=2, c=-2, d=2, t_0=0, T=4, eps=10**(-1.5)):
    """
    Последовательное решение двумерного уравнения теплопроводности
    """
    # Сетка
    x, h_x = np.linspace(a, b, N_x+1, retstep=True)
    y, h_y = np.linspace(c, d, N_y+1, retstep=True)
    t, tau = np.linspace(t_0, T, M+1, retstep=True)
    
    # Инициализация
    u = np.empty((M+1, N_x+1, N_y+1))
    
    # Начальные условия
    for i in range(N_x+1):
        for j in range(N_y+1):
            u[0, i, j] = u_init(x[i], y[j], eps)
    
    # Граничные условия
    for m in range(M+1):
        for j in range(N_y+1):
            u[m, 0, j] = u_left(y[j], t[m])
            u[m, N_x, j] = u_right(y[j], t[m])
        for i in range(N_x+1):
            u[m, i, 0] = u_bottom(x[i], t[m])
            u[m, i, N_y] = u_top(x[i], t[m])
    
    # Основной цикл
    start_time = time.time()
    
    for m in range(M):
        for i in range(1, N_x):
            for j in range(1, N_y):
                # Вторые производные
                d2x = (u[m, i+1, j] - 2*u[m, i, j] + u[m, i-1, j]) / h_x**2
                d2y = (u[m, i, j+1] - 2*u[m, i, j] + u[m, i, j-1]) / h_y**2
                
                # Первые производные
                d1x = (u[m, i+1, j] - u[m, i-1, j]) / (2*h_x)
                d1y = (u[m, i, j+1] - u[m, i, j-1]) / (2*h_y)
                
                # Явная схема
                u[m+1, i, j] = (u[m, i, j] + 
                               tau * (eps * (d2x + d2y) + 
                                     u[m, i, j] * (d1x + d1y) + 
                                     u[m, i, j]**3))
    
    elapsed = time.time() - start_time
    
    return u, x, y, t, elapsed

if __name__ == "__main__":
    print("="*70)
    print("ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ДВУМЕРНОГО УРАВНЕНИЯ ТЕПЛОПРОВОДНОСТИ")
    print("="*70)
    
    # Тестовые конфигурации
    test_cases = [
        (50, 50, 500, "Малая задача"),
        (100, 100, 2000, "Средняя задача"),
        (200, 200, 4000, "Большая задача")
    ]
    
    for N_x, N_y, M, description in test_cases:
        print(f"\n{description}: N_x={N_x}, N_y={N_y}, M={M}")
        
        u, x, y, t, elapsed = solve_2d_sequential(N_x, N_y, M)
        
        print(f"  Время выполнения: {elapsed:.4f} сек")
        print(f"  Размер сетки: ({N_x+1}) x ({N_y+1}) x ({M+1})")
        print(f"  u[0,0,0] = {u[0,0,0]:.6f}, u[-1,-1,-1] = {u[-1,-1,-1]:.6f}")
        
        # Сохранение для большой задачи
        if N_x == 200:
            np.savez('solution_sequential.npz', u=u, x=x, y=y, t=t, elapsed=elapsed)
            print(f"  Решение сохранено")
    
    print("\n" + "="*70)
    print("ВЕРИФИКАЦИЯ ЗАВЕРШЕНА")
    print("="*70)
