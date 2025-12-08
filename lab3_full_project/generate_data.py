import numpy as np

def generate_test_system(M, N, seed=42, well_conditioned=True):
    """
    Генерирует тестовую СЛАУ Ax = b
    
    Parameters:
    M - количество уравнений (строк матрицы)
    N - количество неизвестных (столбцов матрицы)
    seed - seed для генератора случайных чисел
    well_conditioned - если True, создаёт хорошо обусловленную систему
    """
    np.random.seed(seed)
    
    if well_conditioned:
        # Создаём хорошо обусловленную матрицу через SVD
        # A = U * S * V^T, где S - диагональная матрица с управляемыми сингулярными числами
        U = np.linalg.qr(np.random.randn(M, M))[0]
        V = np.linalg.qr(np.random.randn(N, N))[0]
        
        # Сингулярные числа от 1 до 10 (хорошая обусловленность)
        s = np.linspace(10, 1, min(M, N))
        S = np.zeros((M, N))
        np.fill_diagonal(S, s)
        
        A = U @ S @ V.T
    else:
        # Плохо обусловленная матрица (для тестирования регуляризации)
        A = np.random.randn(M, N)
        # Делаем некоторые сингулярные числа очень малыми
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s[-N//4:] = s[-N//4:] * 1e-6  # Последние 25% сингулярных чисел делаем маленькими
        A = U @ np.diag(s) @ Vt
    
    # Создаём истинное решение
    x_true = np.random.randn(N)
    
    # Вычисляем правую часть
    b = A @ x_true
    
    # Добавляем небольшой шум для реалистичности
    b += np.random.randn(M) * 1e-6
    
    return A, b, x_true

def save_system(M, N, filename_prefix='', well_conditioned=True, seed=42):
    """
    Генерирует и сохраняет СЛАУ в файлы
    """
    A, b, x_true = generate_test_system(M, N, seed, well_conditioned)
    
    # Сохраняем размеры
    with open(f'{filename_prefix}in.dat', 'w') as f:
        f.write(f"{M} {N}\n")
    
    # Сохраняем матрицу A
    np.savetxt(f'{filename_prefix}AData.dat', A, fmt='%.10f')
    
    # Сохраняем вектор b
    np.savetxt(f'{filename_prefix}bData.dat', b, fmt='%.10f')
    
    # Сохраняем истинное решение для верификации
    np.savetxt(f'{filename_prefix}x_true.dat', x_true, fmt='%.10f')
    
    # Вычисляем число обусловленности
    cond_number = np.linalg.cond(A)
    
    print(f"Система сгенерирована:")
    print(f"  Размер: {M} x {N}")
    print(f"  Число обусловленности: {cond_number:.2e}")
    print(f"  Норма истинного решения: {np.linalg.norm(x_true):.6f}")
    print(f"  Файлы: {filename_prefix}in.dat, {filename_prefix}AData.dat, {filename_prefix}bData.dat")
    
    return A, b, x_true, cond_number

if __name__ == "__main__":
    print("=" * 70)
    print("ГЕНЕРАЦИЯ ТЕСТОВЫХ СЛАУ")
    print("=" * 70)
    
    # Маленькая система для отладки
    print("\n1. Маленькая хорошо обусловленная система (отладка):")
    print("-" * 70)
    save_system(20, 10, 'small_', well_conditioned=True, seed=42)
    
    # Средняя система
    print("\n2. Средняя хорошо обусловленная система:")
    print("-" * 70)
    save_system(200, 100, 'medium_', well_conditioned=True, seed=42)
    
    # Большая система (основная)
    print("\n3. Большая хорошо обусловленная система (основная):")
    print("-" * 70)
    save_system(1000, 500, '', well_conditioned=True, seed=42)
    
    # Плохо обусловленная система для тестирования регуляризации
    print("\n4. Плохо обусловленная система (для регуляризации):")
    print("-" * 70)
    save_system(1000, 500, 'ill_', well_conditioned=False, seed=42)
    
    print("\n" + "=" * 70)
    print("Генерация завершена!")
    print("=" * 70)
