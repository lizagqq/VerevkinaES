import numpy as np
import time

def sequential_matrix_vector_multiply():
    """
    Последовательное умножение матрицы на вектор
    """
    # Чтение размеров
    with open('in.dat', 'r') as f:
        M, N = map(int, f.readline().split())
    
    print(f"Размеры матрицы: {M} x {N}")
    
    # Чтение матрицы A
    A = np.loadtxt('AData.dat')
    A = A.reshape(M, N)
    
    # Чтение вектора x
    x = np.loadtxt('xData.dat')
    
    # Засекаем время
    start_time = time.time()
    
    # Умножение матрицы на вектор
    b = np.dot(A, x)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Сохранение результата
    np.savetxt('Results_sequential.dat', b, fmt='%.6f')
    
    print(f"Время выполнения: {execution_time:.6f} секунд")
    print(f"Результат сохранён в Results_sequential.dat")
    
    return b, execution_time

if __name__ == "__main__":
    print("=" * 50)
    print("ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ")
    print("=" * 50)
    
    result, exec_time = sequential_matrix_vector_multiply()
    
    print(f"\nПервые 10 элементов результата:")
    print(result[:10])
