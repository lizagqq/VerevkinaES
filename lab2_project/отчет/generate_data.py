import numpy as np

def generate_test_data_part1(M, seed=42):
    """
    Генерирует тестовые данные для Части 1 (скалярное произведение)
    
    Parameters:
    M - длина вектора
    seed - seed для генератора случайных чисел
    """
    np.random.seed(seed)
    
    # Генерируем вектор a
    # Можно использовать arange для простоты верификации
    # a = np.arange(1, M+1, dtype=np.float64)
    
    # Или случайные числа
    a = np.random.rand(M).astype(np.float64)
    
    # Сохраняем вектор в файл
    np.savetxt('vector_a.dat', a, fmt='%.10f')
    
    print(f"Данные для Части 1 сгенерированы:")
    print(f"  - Длина вектора: {M}")
    print(f"  - Файл: vector_a.dat")
    
    return a

def generate_test_data_part2(M, N, seed=42):
    """
    Генерирует тестовые данные для Части 2 (A.T @ x)
    
    Parameters:
    M - количество строк матрицы
    N - количество столбцов матрицы
    seed - seed для генератора случайных чисел
    """
    np.random.seed(seed)
    
    # Генерируем матрицу A размером M x N
    A = np.random.rand(M, N).astype(np.float64)
    
    # Генерируем вектор x длиной M
    x = np.random.rand(M).astype(np.float64)
    
    # Сохраняем размеры в in.dat
    with open('in.dat', 'w') as f:
        f.write(f"{M} {N}\n")
    
    # Сохраняем матрицу A в AData.dat
    np.savetxt('AData.dat', A, fmt='%.10f')
    
    # Сохраняем вектор x в xData.dat
    np.savetxt('xData.dat', x, fmt='%.10f')
    
    print(f"Данные для Части 2 сгенерированы:")
    print(f"  - Размер матрицы: {M} x {N}")
    print(f"  - Файлы: in.dat, AData.dat, xData.dat")
    
    return A, x

if __name__ == "__main__":
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ")
    print("=" * 60)
    
    # Генерируем данные для разных размеров
    
    # Маленькие наборы для отладки
    print("\n1. Маленький набор для отладки:")
    print("-" * 60)
    generate_test_data_part1(10, seed=42)
    generate_test_data_part2(8, 6, seed=42)
    
    # Средние наборы
    print("\n2. Средний набор:")
    print("-" * 60)
    generate_test_data_part1(100, seed=42)
    generate_test_data_part2(100, 80, seed=42)
    
    # Большие наборы для бенчмарков
    print("\n3. Большой набор:")
    print("-" * 60)
    generate_test_data_part1(10000, seed=42)
    generate_test_data_part2(1000, 1000, seed=42)
