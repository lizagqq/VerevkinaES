from mpi4py import MPI
import numpy as np
import time

def parallel_transpose_matrix_vector():
    """
    Параллельное вычисление b = A.T @ x с использованием MPI.Reduce
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
        
        print(f"Размер матрицы: {M} x {N}")
        print(f"Количество процессов: {size}")
        
        # Чтение матрицы A
        A = np.loadtxt('AData.dat', dtype=np.float64)
        A = A.reshape(M, N)
        
        # Чтение вектора x
        x = np.loadtxt('xData.dat', dtype=np.float64)
        
        print(f"Форма матрицы A: {A.shape}")
        print(f"Длина вектора x: {len(x)}")
    
    # Рассылка размеров всем процессам
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    
    # Вычисление распределения строк между процессами
    base_local_M = M // size
    remainder = M % size
    
    # Каждый процесс вычисляет свой размер блока
    if rank < remainder:
        local_M = base_local_M + 1
        start_row = rank * local_M
    else:
        local_M = base_local_M
        start_row = rank * base_local_M + remainder
    
    # Создание буферов для локальных частей
    A_part = np.zeros((local_M, N), dtype=np.float64)
    x_part = np.zeros(local_M, dtype=np.float64)
    
    # Подготовка параметров для Scatterv на процессе 0
    if rank == 0:
        # Вычисление counts и displacements для матрицы A
        sendcounts_A = []
        displacements_A = []
        # Вычисление counts и displacements для вектора x
        sendcounts_x = []
        displacements_x = []
        offset_A = 0
        offset_x = 0
        
        for i in range(size):
            if i < remainder:
                count_rows = base_local_M + 1
            else:
                count_rows = base_local_M
            
            sendcounts_A.append(count_rows * N)
            displacements_A.append(offset_A)
            offset_A += count_rows * N
            
            sendcounts_x.append(count_rows)
            displacements_x.append(offset_x)
            offset_x += count_rows
        
        print(f"Распределение строк: {[c // N for c in sendcounts_A]}")
    else:
        sendcounts_A = None
        displacements_A = None
        sendcounts_x = None
        displacements_x = None
    
    start_time = time.time()
    
    # Распределение матрицы A с помощью Scatterv
    comm.Scatterv([A, sendcounts_A, displacements_A, MPI.DOUBLE], A_part, root=0)
    
    # Распределение вектора x с помощью Scatterv
    comm.Scatterv([x, sendcounts_x, displacements_x, MPI.DOUBLE], x_part, root=0)
    
    # Локальное вычисление: b_temp = A_part.T @ x_part
    # A_part имеет размер (local_M, N)
    # A_part.T имеет размер (N, local_M)
    # x_part имеет размер (local_M,)
    # Результат b_temp имеет размер (N,)
    b_temp = np.dot(A_part.T, x_part)
    
    # Глобальная редукция: суммирование векторов b_temp со всех процессов
    if rank == 0:
        b = np.zeros(N, dtype=np.float64)
    else:
        b = None
    
    # Используем Reduce с операцией SUM для суммирования векторов
    comm.Reduce(b_temp, b, op=MPI.SUM, root=0)
    
    end_time = time.time()
    
    if rank == 0:
        execution_time = end_time - start_time
        
        # Верификация - последовательное вычисление
        seq_start = time.time()
        b_seq = np.dot(A.T, x)
        seq_end = time.time()
        seq_time = seq_end - seq_start
        
        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТЫ")
        print(f"{'='*60}")
        
        # Вычисление ошибок
        abs_error = np.max(np.abs(b - b_seq))
        rel_error = abs_error / np.max(np.abs(b_seq)) if np.max(np.abs(b_seq)) > 0 else 0
        
        print(f"Максимальная абсолютная ошибка: {abs_error:.2e}")
        print(f"Максимальная относительная ошибка: {rel_error:.2e}")
        print(f"\nПервые 10 элементов результата (параллельный): {b[:10]}")
        print(f"Первые 10 элементов результата (последовательный): {b_seq[:10]}")
        print(f"\nВремя параллельного выполнения: {execution_time:.6f} сек")
        print(f"Время последовательного выполнения: {seq_time:.6f} сек")
        print(f"Ускорение: {seq_time / execution_time:.2f}x")
        
        # Сохранение результата
        np.savetxt('Results_part2_parallel.dat', b, fmt='%.10f')
        print(f"\nРезультат сохранён в Results_part2_parallel.dat")
        
        return b, execution_time
    
    return None, None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 60)
        print("ЧАСТЬ 2: УМНОЖЕНИЕ A.T @ x")
        print("=" * 60)
    
    result, exec_time = parallel_transpose_matrix_vector()
