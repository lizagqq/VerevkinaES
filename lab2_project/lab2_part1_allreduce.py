from mpi4py import MPI
import numpy as np
import time

def parallel_dot_product_allreduce():
    """
    Параллельное вычисление скалярного произведения с использованием MPI.Allreduce
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Инициализация переменных
    a = None
    M = None
    
    # Процесс 0 создает/загружает вектор
    if rank == 0:
        # Загрузка из файла
        a = np.loadtxt('vector_a.dat', dtype=np.float64)
        M = len(a)
        
        print(f"Длина вектора: {M}")
        print(f"Количество процессов: {size}")
    
    # Рассылка размера вектора всем процессам
    M = comm.bcast(M, root=0)
    
    # Вычисление распределения элементов между процессами
    base_local_M = M // size
    remainder = M % size
    
    # Каждый процесс вычисляет свой размер блока
    if rank < remainder:
        local_M = base_local_M + 1
        start_idx = rank * local_M
    else:
        local_M = base_local_M
        start_idx = rank * base_local_M + remainder
    
    # Создание буфера для локальной части вектора
    a_part = np.zeros(local_M, dtype=np.float64)
    
    # Подготовка параметров для Scatterv на процессе 0
    if rank == 0:
        # Вычисление counts и displacements
        sendcounts = []
        displacements = []
        offset = 0
        
        for i in range(size):
            if i < remainder:
                count = base_local_M + 1
            else:
                count = base_local_M
            sendcounts.append(count)
            displacements.append(offset)
            offset += count
    else:
        sendcounts = None
        displacements = None
    
    start_time = time.time()
    
    # Распределение вектора с помощью Scatterv
    comm.Scatterv([a, sendcounts, displacements, MPI.DOUBLE], a_part, root=0)
    
    # Локальное вычисление скалярного произведения
    local_dot = np.dot(a_part, a_part)
    
    # Глобальная редукция с помощью Allreduce (результат на всех процессах)
    global_dot = np.zeros(1, dtype=np.float64)
    comm.Allreduce(np.array([local_dot], dtype=np.float64), global_dot, op=MPI.SUM)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Все процессы имеют результат
    if rank == 0:
        # Верификация - последовательное вычисление
        seq_start = time.time()
        seq_dot = np.dot(a, a)
        seq_end = time.time()
        seq_time = seq_end - seq_start
        
        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТЫ (MPI.Allreduce)")
        print(f"{'='*60}")
        print(f"Параллельное скалярное произведение: {global_dot[0]:.10f}")
        print(f"Последовательное скалярное произведение: {seq_dot:.10f}")
        print(f"Абсолютная ошибка: {abs(global_dot[0] - seq_dot):.2e}")
        print(f"Относительная ошибка: {abs(global_dot[0] - seq_dot) / abs(seq_dot):.2e}")
        print(f"\nВремя параллельного выполнения: {execution_time:.6f} сек")
        print(f"Время последовательного выполнения: {seq_time:.6f} сек")
        print(f"Ускорение: {seq_time / execution_time:.2f}x")
        
        return global_dot[0], execution_time
    
    # На других процессах также есть результат
    return global_dot[0], execution_time

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 60)
        print("ЧАСТЬ 1: СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ (MPI.Allreduce)")
        print("=" * 60)
    
    result, exec_time = parallel_dot_product_allreduce()
    
    # Все процессы могут вывести результат
    print(f"Процесс {rank}: результат = {result:.10f}, время = {exec_time:.6f} сек")
