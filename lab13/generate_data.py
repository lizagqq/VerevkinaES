import numpy as np
import argparse

def generate_test_data(M, N, condition_number=10, seed=42):
    """
    Генерирует тестовые данные для системы уравнений Ax = b
    
    Parameters:
    M - число строк матрицы A
    N - число столбцов матрицы A
    condition_number - число обусловленности матрицы
    seed - seed для генератора случайных чисел
    """
    np.random.seed(seed)
    
    # Создаём матрицу с заданным числом обусловленности
    # Используем SVD: A = U * Sigma * V^T
    U = np.linalg.qr(np.random.randn(M, M))[0]
    V = np.linalg.qr(np.random.randn(N, N))[0]
    
    # Создаём диагональную матрицу с заданным числом обусловленности
    sigma_max = 1.0
    sigma_min = sigma_max / condition_number
    singular_values = np.linspace(sigma_max, sigma_min, min(M, N))
    
    Sigma = np.zeros((M, N))
    np.fill_diagonal(Sigma, singular_values)
    
    # A = U * Sigma * V^T
    A = U @ Sigma @ V.T
    
    # Генерируем истинное решение
    x_true = np.random.randn(N)
    
    # Вычисляем b = A @ x_true
    b = A @ x_true
    
    # Добавляем небольшой шум к b
    noise_level = 1e-10
    b += noise_level * np.random.randn(M)
    
    return A, b, x_true

def save_test_data(prefix, M, N, condition_number=10):
    """
    Генерирует и сохраняет тестовые данные
    """
    A, b, x_true = generate_test_data(M, N, condition_number)
    
    # Сохраняем размеры
    with open(f'{prefix}in.dat', 'w') as f:
        f.write(f'{M} {N}\n')
    
    # Сохраняем данные
    np.savetxt(f'{prefix}AData.dat', A.flatten(), fmt='%.15e')
    np.savetxt(f'{prefix}bData.dat', b, fmt='%.15e')
    np.savetxt(f'{prefix}x_true.dat', x_true, fmt='%.15e')
    
    print(f"Данные сохранены с префиксом '{prefix}':")
    print(f"  Размер системы: {M} x {N}")
    print(f"  Число обусловленности: {np.linalg.cond(A):.2e}")
    print(f"  Норма ||A||: {np.linalg.norm(A):.2e}")
    print(f"  Норма ||b||: {np.linalg.norm(b):.2e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Генерация тестовых данных')
    parser.add_argument('--size', type=int, nargs=2, default=[1000, 500],
                        help='Размеры матрицы M N')
    parser.add_argument('--condition', type=float, default=10,
                        help='Число обусловленности')
    parser.add_argument('--prefix', type=str, default='',
                        help='Префикс для имён файлов')
    
    args = parser.parse_args()
    
    save_test_data(args.prefix, args.size[0], args.size[1], args.condition)
