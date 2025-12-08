#!/usr/bin/env python3
"""
Лабораторная работа №5: Двумерная декомпозиция матрицы
Умножение матрицы на вектор с разбиением на блоки
"""

from mpi4py import MPI
import numpy as np
import sys

def auxiliary_arrays_determination(M, num):
    """
    Определяет количество элементов и смещения для распределения
    массива длины M между num процессами
    """
    base_count = M // num
    remainder = M % num
    
    rcounts = np.array([base_count + (1 if i < remainder else 0) for i in range(num)], dtype=np.int32)
    displs = np.zeros(num, dtype=np.int32)
    
    for i in range(1, num):
        displs[i] = displs[i-1] + rcounts[i-1]
    
    return rcounts, displs

def matvec_2d(comm, A_part, x_part, M, N, M_part, N_part, comm_row, comm_col):
    """
    Умножение матрицы на вектор с двумерной декомпозицией
    
    A_part - блок матрицы на данном процессе (M_part x N_part)
    x_part - часть вектора на данном процессе (N_part)
    """
    rank = comm.Get_rank()
    
    # Локальное умножение A_part @ x_part
    b_part_temp = np.dot(A_part, x_part)
    
    # Редукция вдоль строк (суммирование по столбцам сетки)
    # Результат на процессах первого столбца (rank % num_col == 0)
    b_part = np.zeros_like(b_part_temp) if comm_row.Get_rank() == 0 else None
    comm_row.Reduce(b_part_temp, b_part, op=MPI.SUM, root=0)
    
    return b_part

def main():
    # Инициализация MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Проверка: numprocs должно быть полным квадратом
    sqrt_size = int(np.sqrt(size))
    if sqrt_size * sqrt_size != size:
        if rank == 0:
            print(f"ОШИБКА: Количество процессов ({size}) должно быть полным квадратом!")
            print(f"Допустимые значения: 1, 4, 9, 16, 25, 36, 49, 64, ...")
        sys.exit(1)
    
    num_row = num_col = sqrt_size
    
    if rank == 0:
        print("="*70)
        print("УМНОЖЕНИЕ МАТРИЦЫ НА ВЕКТОР С ДВУМЕРНОЙ ДЕКОМПОЗИЦИЕЙ")
        print("="*70)
        print(f"Сетка процессов: {num_row} × {num_col} = {size} процессов")
    
    # Создание коммуникаторов для строк и столбцов
    col_color = rank % num_col  # Процессы в одном столбце
    row_color = rank // num_col  # Процессы в одной строке
    
    comm_col = comm.Split(color=col_color, key=rank)
    comm_row = comm.Split(color=row_color, key=rank)
    
    row_rank = comm_row.Get_rank()
    col_rank = comm_col.Get_rank()
    
    # Чтение размеров матрицы
    if rank == 0:
        try:
            with open('in.dat', 'r') as f:
                M, N = map(int, f.read().strip().split())
            print(f"\nРазмер матрицы: {M} × {N}")
        except FileNotFoundError:
            print("ОШИБКА: Файл in.dat не найден!")
            M, N = 0, 0
    else:
        M, N = 0, 0
    
    # Рассылка размеров
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    
    if M == 0 or N == 0:
        sys.exit(1)
    
    # Определение размеров блоков
    if rank == 0:
        rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
        rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)
    else:
        rcounts_M, displs_M = None, None
        rcounts_N, displs_N = None, None
    
    # Распределение размеров блоков
    # Шаг 1: Scatter rcounts_N по первой строке (row_rank == 0)
    if row_rank == 0:
        N_part = np.zeros(1, dtype=np.int32)
        comm_row.Scatter(rcounts_N if rank == 0 else None, N_part, root=0)
        N_part = N_part[0]
    else:
        N_part = 0
    
    # Шаг 2: Bcast N_part внутри каждого столбца
    N_part = comm_col.bcast(N_part, root=0)
    
    # Аналогично для M_part
    if col_rank == 0:
        M_part = np.zeros(1, dtype=np.int32)
        comm_col.Scatter(rcounts_M if rank == 0 else None, M_part, root=0)
        M_part = M_part[0]
    else:
        M_part = 0
    
    M_part = comm_row.bcast(M_part, root=0)
    
    # Чтение данных на процессе 0
    if rank == 0:
        try:
            A = np.loadtxt('AData.dat', dtype=np.float64)
            if A.ndim == 1:
                A = A.reshape(M, N)
            x = np.loadtxt('xData.dat', dtype=np.float64)
            print(f"Данные загружены успешно")
        except Exception as e:
            print(f"ОШИБКА при чтении данных: {e}")
            A = np.random.randn(M, N)
            x = np.random.randn(N)
            print(f"Используются случайные данные")
    else:
        A = None
        x = None
    
    # Распределение матрицы A (упрощённая версия через Scatterv)
    A_part = np.zeros((M_part, N_part), dtype=np.float64)
    
    # Для каждой строки сетки процессов распределяем данные
    if col_rank == 0:
        # Процессы первого столбца получают данные от root
        if rank == 0:
            # Подготовка данных для рассылки по строкам
            A_rows = [A[displs_M[i]:displs_M[i]+rcounts_M[i], :] for i in range(num_row)]
        else:
            A_rows = None
        
        # Scatter блоков строк
        A_row_block = comm_col.scatter(A_rows, root=0)
    else:
        A_row_block = None
    
    # Теперь распределяем по столбцам внутри каждой строки
    if col_rank == 0 and A_row_block is not None:
        # Подготовка данных для рассылки по столбцам
        A_cols = [A_row_block[:, displs_N[i]:displs_N[i]+rcounts_N[i]] for i in range(num_col)]
    else:
        A_cols = None
    
    A_part = comm_row.scatter(A_cols, root=0)
    
    # Распределение вектора x
    if row_rank == 0:
        # Scatterv по первой строке
        x_part = np.zeros(N_part, dtype=np.float64)
        
        if rank == 0:
            sendbuf = [x[displs_N[i]:displs_N[i]+rcounts_N[i]] for i in range(num_col)]
        else:
            sendbuf = None
        
        x_part = comm_row.scatter(sendbuf, root=0)
    else:
        x_part = np.zeros(N_part, dtype=np.float64)
    
    # Bcast x_part внутри каждого столбца
    x_part = comm_col.bcast(x_part, root=0)
    
    # Измерение времени
    comm.Barrier()
    start_time = MPI.Wtime()
    
    # Умножение матрицы на вектор
    b_part = matvec_2d(comm, A_part, x_part, M, N, M_part, N_part, comm_row, comm_col)
    
    comm.Barrier()
    end_time = MPI.Wtime()
    
    # Сбор результата на процессе 0
    if col_rank == 0:
        # Gatherv результатов на процессе 0
        if rank == 0:
            b = np.zeros(M, dtype=np.float64)
            recvbuf = [b, (rcounts_M, displs_M)]
        else:
            recvbuf = None
        
        comm_col.Gatherv(b_part, recvbuf, root=0)
    
    # Вывод результатов
    if rank == 0:
        elapsed_time = end_time - start_time
        print(f"\nВремя выполнения: {elapsed_time:.6f} сек")
        
        # Проверка корректности (если были считаны реальные данные)
        if A is not None and x is not None:
            b_ref = np.dot(A, x)
            error = np.linalg.norm(b - b_ref) / np.linalg.norm(b_ref)
            print(f"Относительная ошибка: {error:.2e}")
        
        # Сохранение результата
        np.savetxt('bData_2d.dat', b, fmt='%.10f')
        print(f"Результат сохранён в bData_2d.dat")
    
    # Освобождение коммуникаторов
    comm_row.Free()
    comm_col.Free()

if __name__ == "__main__":
    main()
