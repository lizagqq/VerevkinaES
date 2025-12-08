#!/usr/bin/env python3
"""
Последовательная версия неявной схемы ROS1
"""
import numpy as np
import time

def u_init(x):
    """Начальное условие"""
    return np.sin(3 * np.pi * (x - 1/6))

def u_left(t):
    """Граничное условие слева"""
    return -1

def u_right(t):
    """Граничное условие справа"""
    return +1

def f(y, t, h, N, u_left_val, u_right_val, eps):
    """
    Вычисление правой части системы
    """
    f_vec = np.empty(N - 1, dtype=np.float64)
    
    # Первый элемент
    f_vec[0] = (eps * (y[1] - 2*y[0] + u_left_val) / h**2 + 
                y[0] * (y[1] - u_left_val) / (2*h) + 
                y[0]**3)
    
    # Внутренние элементы
    for n in range(1, N - 2):
        f_vec[n] = (eps * (y[n+1] - 2*y[n] + y[n-1]) / h**2 + 
                    y[n] * (y[n+1] - y[n-1]) / (2*h) + 
                    y[n]**3)
    
    # Последний элемент
    f_vec[N-2] = (eps * (u_right_val - 2*y[N-2] + y[N-3]) / h**2 + 
                  y[N-2] * (u_right_val - y[N-3]) / (2*h) + 
                  y[N-2]**3)
    
    return f_vec

def diagonal_preparation(y, t, h, N, u_left_val, u_right_val, eps, tau, alpha):
    """
    Формирование трёхдиагональной матрицы
    """
    a = np.empty(N-1, dtype=np.float64)
    b = np.empty(N-1, dtype=np.float64)
    c = np.empty(N-1, dtype=np.float64)
    
    # Первая строка
    b[0] = 1.0 - alpha*tau*(-2*eps/h**2 + (y[1] - u_left_val)/(2*h) + 3*y[0]**2)
    c[0] = -alpha * tau * (eps/h**2 + y[0]/(2*h))
    
    # Внутренние строки
    for n in range(1, N-2):
        a[n] = -alpha*tau*(eps/h**2 - y[n]/(2*h))
        b[n] = 1.0 - alpha*tau*(-2*eps/h**2 + (y[n+1] - y[n-1])/(2*h) + 3*y[n]**2)
        c[n] = -alpha*tau*(eps/h**2 + y[n]/(2*h))
    
    # Последняя строка
    a[N-2] = -alpha*tau*(eps/h**2 - y[N-2]/(2*h))
    b[N-2] = 1.0 - alpha*tau*(-2*eps/h**2 + (u_right_val - y[N-3])/(2*h) + 3*y[N-2]**2)
    
    return a, b, c

def tridiagonal_solve(a, b, c, d):
    """
    Решение трёхдиагональной системы методом прогонки
    """
    N = len(d)
    x = np.empty(N, dtype=np.float64)
    
    # Прямой ход
    for n in range(1, N):
        coef = a[n] / b[n-1]
        b[n] = b[n] - coef * c[n-1]
        d[n] = d[n] - coef * d[n-1]
    
    # Обратный ход
    x[N-1] = d[N-1] / b[N-1]
    for n in range(N-2, -1, -1):
        x[n] = (d[n] - c[n] * x[n+1]) / b[n]
    
    return x

def solve_sequential(N, M, a=0, b=1, t_0=0, T=2.0, eps=10**(-1.5), alpha=0.5):
    """
    Последовательное решение неявной схемой ROS1
    """
    # Сетка
    x, h = np.linspace(a, b, N+1, retstep=True)
    t, tau = np.linspace(t_0, T, M+1, retstep=True)
    
    # Инициализация
    y = np.array([u_init(x[i]) for i in range(1, N)])
    u = np.empty((M+1, N+1))
    
    # Начальные условия
    for n in range(N+1):
        u[0, n] = u_init(x[n])
    
    # Граничные условия
    for m in range(M+1):
        u[m, 0] = u_left(t[m])
        u[m, -1] = u_right(t[m])
    
    # Основной цикл
    start_time = time.time()
    
    for m in range(M):
        u_left_val = u_left(t[m])
        u_right_val = u_right(t[m])
        
        # Формирование матрицы
        a_diag, b_diag, c_diag = diagonal_preparation(
            y, t[m], h, N, u_left_val, u_right_val, eps, tau, alpha)
        
        # Правая часть
        rhs = f(y, t[m] + tau/2, h, N, u_left_val, u_right_val, eps)
        
        # Решение СЛАУ
        w_1 = tridiagonal_solve(a_diag.copy(), b_diag.copy(), c_diag.copy(), rhs)
        
        # Обновление решения
        y = y + tau * np.real(w_1)
        u[m+1, 1:-1] = y
    
    elapsed = time.time() - start_time
    
    return u, x, t, elapsed

if __name__ == "__main__":
    print("="*70)
    print("ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ НЕЯВНОЙ СХЕМЫ ROS1")
    print("="*70)
    
    # Тестовые конфигурации
    test_cases = [
        (200, 300, "Малая задача"),
        (2000, 1000, "Средняя задача"),
        (20000, 5000, "Большая задача")
    ]
    
    for N, M, description in test_cases:
        print(f"\n{description}: N={N}, M={M}")
        
        u, x, t, elapsed = solve_sequential(N, M)
        
        print(f"  Время выполнения: {elapsed:.4f} сек")
        print(f"  Размер сетки: {N+1} x {M+1}")
        print(f"  u[0,0] = {u[0,0]:.6f}, u[-1,-1] = {u[-1,-1]:.6f}")
        
        # Сохранение для большой задачи
        if N == 20000:
            np.savez('solution_sequential.npz', u=u, x=x, t=t, elapsed=elapsed)
            print(f"  Решение сохранено")
    
    print("\n" + "="*70)
    print("ВЕРИФИКАЦИЯ ЗАВЕРШЕНА")
    print("="*70)
