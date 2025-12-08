import numpy as np

def generate_test_data(M, N, seed=42):
    """
    Генерирует тестовые данные для умножения матрицы на вектор
    
    Parameters:
    M - количество строк матрицы
    N - количество столбцов матрицы (и длина вектора)
    seed - seed для генератора случайных чисел
    """
    np.random.seed(seed)
    
    # Генерируем матрицу A размером M x N
    A = np.random.rand(M, N)
    
    # Генерируем вектор x длиной N
    x = np.random.rand(N)
    
    # Сохраняем размеры в in.dat
    with open('in.dat', 'w') as f:
        f.write(f"{M} {N}\n")
    
    # Сохраняем матрицу A в AData.dat
    np.savetxt('AData.dat', A, fmt='%.6f')
    
    # Сохраняем вектор x в xData.dat
    np.savetxt('xData.dat', x, fmt='%.6f')
    
    print(f"Тестовые данные сгенерированы:")
    print(f"  - Размер матрицы: {M} x {N}")
    print(f"  - Файлы: in.dat, AData.dat, xData.dat")

if __name__ == "__main__":
    # Генерируем несколько наборов данных для тестирования
    
    # Маленький набор для отладки
    print("Генерация маленького набора данных (8x6)...")
    generate_test_data(8, 6, seed=42)
    
    # Средний набор
    print("\nГенерация среднего набора данных (100x80)...")
    generate_test_data(100, 80, seed=42)
    
    # Большой набор для бенчмарков
    print("\nГенерация большого набора данных (1000x1000)...")
    generate_test_data(1000, 1000, seed=42)
