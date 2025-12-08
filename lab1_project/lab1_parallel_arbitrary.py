from mpi4py import MPI
import numpy as np
import time

def parallel_matrix_vector_multiply_arbitrary():
    """
    Параллельное умножение матрицы на вектор с поддержкой произвольного размера
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
    
    # Вычисление распределения строк между процессами
    base_local_M = M // size
    remainder = M % size
    
    # Каждый процесс вычисляет свой размер блока и смещение
    if rank < remainder:
        local_M = base_local_M + 1
        start_row = rank * local_M
    else:
        local_M = base_local_M
        start_row = rank * base_local_M + remainder
    
    # Создание буфера для локальной части матрицы
    A_part = np.zeros((local_M, N), dtype=np.float64)
    
    start_time = time.time()
    
    # Подготовка данных для Scatterv на процессе 0
    if rank == 0:
        # Количество элементов для каждого процесса
        sendcounts = []
        displacements = []
        offset = 0
        
        for i in range(size):
            if i < remainder:
                count = (base_local_M + 1) * N
            else:
                count = base_local_M * N
            sendcounts.append(count)
            displacements.append(offset)
            offset += count
        
        print(f"Распределение строк: {[c // N for c in sendcounts]}")
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
        recvcounts = []
        displacements_recv = []
        offset = 0
        
        for i in range(size):
            if i < remainder:
                count = base_local_M + 1
            else:
                count = base_local_M
            recvcounts.append(count)
            displacements_recv.append(offset)
            offset += count
    else:
        b = None
        recvcounts = None
        displacements_recv = None
    
    comm.Gatherv(b_part, [b, recvcounts, displacements_recv, MPI.DOUBLE], root=0)
    
    if rank == 0:
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Сохранение результата
        np.savetxt('Results_parallel_arbitrary.dat', b, fmt='%.6f')
        
        print(f"Время выполнения: {execution_time:.6f} секунд")
        print(f"Результат сохранён в Results_parallel_arbitrary.dat")
        
        return b, execution_time
    else:
        return None, None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 50)
        print("ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (Произвольный размер)")
        print("=" * 50)
    
    result, exec_time = parallel_matrix_vector_multiply_arbitrary()
    
    if rank == 0:
        print(f"\nПервые 10 элементов результата:")
        print(result[:10])
