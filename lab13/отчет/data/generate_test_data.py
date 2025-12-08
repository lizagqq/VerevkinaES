"""
Генератор тестовых данных для метода сопряженных градиентов
"""
import numpy as np

def generate_test_system(M, N, condition_number=10.0, seed=42):
    """
    Генерирует переопределённую систему Ax = b с известным решением
    
    Parameters:
    M - количество уравнений (строк)
    N - количество неизвестных (столбцов)
    condition_number - число обусловленности матрицы
    seed - seed для воспроизводимости
    """
    np.random.seed(seed)
    
    # Генерируем истинное решение
    x_true = np.random.randn(N)
    
    # Генерируем хорошо обусловленную матрицу через SVD
    U = np.linalg.qr(np.random.randn(M, M))[0]
    V = np.linalg.qr(np.random.randn(N, N))[0]
    
    # Создаём сингулярные значения с заданным числом обусловленности
    s_max = 1.0
    s_min = s_max / condition_number
    singular_values = np.linspace(s_max, s_min, min(M, N))
    
    # Формируем диагональную матрицу
    S = np.zeros((M, N))
    np.fill_diagonal(S, singular_values)
    
    # A = U @ S @ V^T
    A = U @ S @ V.T
    
    # b = A @ x_true
    b = A @ x_true
    
    return A, b, x_true

if __name__ == "__main__":
    # Генерируем данные разных размеров
    test_sizes = [
        (500, 100),   # Малая
        (1000, 200),  # Средняя
        (2000, 400),  # Большая
    ]
    
    for M, N in test_sizes:
        print(f"Генерация системы {M}x{N}...")
        A, b, x_true = generate_test_system(M, N)
        
        # Сохраняем данные
        prefix = f"test_{M}x{N}"
        
        with open(f'{prefix}_in.dat', 'w') as f:
            f.write(f"{M} {N}\n")
        
        np.savetxt(f'{prefix}_AData.dat', A.ravel(), fmt='%.10f')
        np.savetxt(f'{prefix}_bData.dat', b, fmt='%.10f')
        np.savetxt(f'{prefix}_x_true.dat', x_true, fmt='%.10f')
        
        print(f"  Число обусловленности: {np.linalg.cond(A):.2e}")
        print(f"  Сохранено в {prefix}_*.dat")
    
    print("\nГотово!")
