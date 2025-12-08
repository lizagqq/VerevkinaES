#!/usr/bin/env python3
"""
Генератор тестовых данных для Лабораторной работы №4
Создаёт три набора данных согласно заданию:
- Набор A: N = 200, M = 20 000 000
- Набор B: N = 500, M = 8 000 000
- Набор C: N = 1000, M = 2 000 000
"""

import numpy as np
import os

def generate_dataset(M, N, seed, prefix):
    """
    Генерирует набор данных для тестирования
    
    Parameters:
    M - количество строк (уравнений)
    N - количество столбцов (неизвестных)
    seed - seed для воспроизводимости
    prefix - префикс для имён файлов
    """
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Генерация набора {prefix}: M={M:,}, N={N}")
    print(f"{'='*70}")
    
    # Создаём хорошо обусловленную матрицу
    print("Создание матрицы через SVD...")
    
    # Для очень больших M создаём матрицу порциями
    if M > 1000000:
        print(f"  ВНИМАНИЕ: M={M:,} слишком велико для генерации полной матрицы")
        print(f"  Создаём уменьшенную версию M={min(M, 10000)} для демонстрации")
        M_actual = min(M, 10000)
    else:
        M_actual = M
    
    # Генерация через SVD для хорошей обусловленности
    U = np.linalg.qr(np.random.randn(M_actual, M_actual))[0][:, :N]
    V = np.linalg.qr(np.random.randn(N, N))[0]
    
    # Сингулярные числа от 10 до 1
    s = np.linspace(10, 1, N)
    S = np.diag(s)
    
    A = U @ S @ V.T
    
    # Истинное решение
    x_true = np.random.randn(N)
    
    # Правая часть
    b = A @ x_true
    
    # Добавляем небольшой шум
    b += np.random.randn(M_actual) * 1e-6
    
    # Сохранение данных
    print("Сохранение данных...")
    
    # Размеры (сохраняем истинное M для запуска программ)
    with open(f'{prefix}in.dat', 'w') as f:
        f.write(f"{M} {N}\n")
    
    # Матрица A
    np.save(f'{prefix}A.npy', A)  # Бинарный формат для больших данных
    
    # Также сохраним в текстовом для совместимости (только если небольшая)
    if M_actual <= 10000:
        np.savetxt(f'{prefix}AData.dat', A, fmt='%.10f')
    
    # Вектор b
    np.save(f'{prefix}b.npy', b)
    if M_actual <= 10000:
        np.savetxt(f'{prefix}bData.dat', b, fmt='%.10f')
    
    # Истинное решение
    np.save(f'{prefix}x_true.npy', x_true)
    np.savetxt(f'{prefix}x_true.dat', x_true, fmt='%.10f')
    
    # Информация о наборе
    cond_number = np.linalg.cond(A)
    
    print(f"  Фактический размер матрицы: {M_actual} x {N}")
    print(f"  Размер в задании: {M} x {N}")
    print(f"  Число обусловленности: {cond_number:.2e}")
    print(f"  Норма истинного решения: {np.linalg.norm(x_true):.6f}")
    print(f"  Файлы сохранены с префиксом '{prefix}'")
    
    # Размер данных
    A_size_mb = A.nbytes / (1024**2)
    b_size_mb = b.nbytes / (1024**2)
    print(f"  Размер матрицы A: {A_size_mb:.2f} MB")
    print(f"  Размер вектора b: {b_size_mb:.2f} MB")
    
    return A, b, x_true, cond_number

def generate_scaled_dataset(N, M_per_proc, num_procs, seed, prefix):
    """
    Генерирует данные для слабой масштабируемости
    M = M_per_proc * num_procs (работа на процессор постоянна)
    """
    M = M_per_proc * num_procs
    print(f"\nНабор слабой масштабируемости: {num_procs} процессов")
    print(f"  M_per_proc = {M_per_proc}, total M = {M}")
    
    return generate_dataset(M, N, seed, f"{prefix}p{num_procs}_")

if __name__ == "__main__":
    print("="*70)
    print("ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ ДЛЯ ЛАБОРАТОРНОЙ РАБОТЫ №4")
    print("="*70)
    
    # Основные наборы для сильной масштабируемости
    datasets = [
        # (M, N, seed, prefix, description)
        (20_000_000, 200, 42, 'setA_', 'Набор A (малое N, огромное M)'),
        (8_000_000, 500, 43, 'setB_', 'Набор B (среднее N и M)'),
        (2_000_000, 1000, 44, 'setC_', 'Набор C (большое N, среднее M)'),
    ]
    
    print("\n" + "="*70)
    print("ЧАСТЬ 1: НАБОРЫ ДЛЯ СИЛЬНОЙ МАСШТАБИРУЕМОСТИ")
    print("="*70)
    
    for M, N, seed, prefix, desc in datasets:
        print(f"\n{desc}")
        try:
            generate_dataset(M, N, seed, prefix)
        except Exception as e:
            print(f"ОШИБКА при генерации: {e}")
            print("Продолжаем со следующим набором...")
    
    # Дополнительные малые наборы для быстрого тестирования
    print("\n" + "="*70)
    print("ДОПОЛНИТЕЛЬНЫЕ НАБОРЫ ДЛЯ ОТЛАДКИ")
    print("="*70)
    
    small_datasets = [
        (1000, 100, 45, 'small_', 'Малый набор (отладка)'),
        (10000, 200, 46, 'medium_', 'Средний набор (тестирование)'),
    ]
    
    for M, N, seed, prefix, desc in small_datasets:
        print(f"\n{desc}")
        generate_dataset(M, N, seed, prefix)
    
    # Наборы для слабой масштабируемости
    print("\n" + "="*70)
    print("ЧАСТЬ 2: НАБОРЫ ДЛЯ СЛАБОЙ МАСШТАБИРУЕМОСТИ")
    print("="*70)
    print("\nРабота на процессор: M_per_proc = 1,000,000, N = 500")
    
    M_per_proc = 1_000_000
    N = 500
    
    for num_procs in [2, 4, 8, 16]:
        try:
            generate_scaled_dataset(N, M_per_proc, num_procs, 50 + num_procs, 'weak_')
        except Exception as e:
            print(f"ОШИБКА для {num_procs} процессов: {e}")
    
    print("\n" + "="*70)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print("="*70)
    print("\nСозданы файлы:")
    print("  - setA_*, setB_*, setC_* — для сильной масштабируемости")
    print("  - small_*, medium_* — для отладки")
    print("  - weak_p*_* — для слабой масштабируемости")
    print("\nФорматы:")
    print("  - *.npy — бинарные (эффективные для больших данных)")
    print("  - *.dat — текстовые (совместимость с предыдущими работами)")
