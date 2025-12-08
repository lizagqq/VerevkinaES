from mpi4py import MPI
import numpy as np
import time

def parallel_matrix_vector_multiply_collective():
    """
    Параллельное умножение матрицы на вектор с использованием коллективных операций
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Инициализация переменных
    M = None
    N = None
    A = None
    x = None
    
    # Процесс 0 читает данные
    if rank == 0:
        # Чтение размеров
        with open('in.dat', 'r') as f:
            M, N = map(int, f.readline().split())
        
        print(f"Размеры матрицы: {M} x {N}")
        print(f"Количество процессов: {size}")
        
        # Чтение матрицы A
        A = np.loadtxt('AData.dat')
        A = A.reshape(M, N)
        
        # Чтение вектора x
        x = np.loadtxt('xData.dat')
    
    # Рассылка размеров всем процессам
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    
    # Рассылка вектора x всем процессам
    x = comm.bcast(x, root=0)
    
    # Вычисление размера локального блока
    local_M = M // size
    
    # Создание буфера для локальной части матрицы
    A_part = np.zeros((local_M, N), dtype=np.float64)
    
    start_time = time.time()
    
    # Подготовка данных для Scatterv на процессе 0
    if rank == 0:
        # Количество элементов для каждого процесса
        sendcounts = [local_M * N for _ in range(size)]
        # Смещения для каждого процесса
        displacements = [i * local_M * N for i in range(size)]
    else:
        sendcounts = None
        displacements = None
    
    # Распределение блоков матрицы с помощью Scatterv
    comm.Scatterv([A, sendcounts, displacements, MPI.DOUBLE], A_part, root=0)
    
    # Локальное вычисление
    b_part = np.dot(A_part, x)
    
    # Сбор результатов на процессе 0 с помощью Gatherv
    if rank == 0:
        b = np.zeros(M, dtype=np.float64)
        recvcounts = [local_M for _ in range(size)]
        displacements_recv = [i * local_M for i in range(size)]
    else:
        b = None
        recvcounts = None
        displacements_recv = None
    
    comm.Gatherv(b_part, [b, recvcounts, displacements_recv, MPI.DOUBLE], root=0)
    
    if rank == 0:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Сохранение результата
        np.savetxt('Results_parallel_collective.dat', b, fmt='%.6f')
        
        print(f"Время выполнения: {execution_time:.6f} секунд")
        print(f"Результат сохранён в Results_parallel_collective.dat")
        
        return b, execution_time
    else:
        return None, None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 50)
        print("ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (Коллективные операции)")
        print("=" * 50)
    
    result, exec_time = parallel_matrix_vector_multiply_collective()
    
    if rank == 0:
        print(f"\nПервые 10 элементов результата:")
        print(result[:10])
