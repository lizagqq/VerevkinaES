from mpi4py import MPI
import numpy as np
import time

def parallel_matrix_vector_multiply_send_recv():
    """
    Параллельное умножение матрицы на вектор с использованием Send/Recv
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
        
        # Проверка делимости M на size
        if M % size != 0:
            print(f"ВНИМАНИЕ: M={M} не делится на size={size} без остатка!")
            print("Для этой версии используйте M, кратное числу процессов.")
        
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
    
    # Процесс 0 распределяет данные
    if rank == 0:
        # Процесс 0 оставляет себе первый блок
        A_part = A[0:local_M, :].copy()
        
        # Отправка блоков другим процессам
        for i in range(1, size):
            start_row = i * local_M
            end_row = start_row + local_M
            comm.Send(A[start_row:end_row, :].copy(), dest=i, tag=i)
    else:
        # Остальные процессы получают свои блоки
        comm.Recv(A_part, source=0, tag=rank)
    
    # Локальное вычисление
    b_part = np.dot(A_part, x)
    
    # Сбор результатов на процессе 0
    if rank == 0:
        # Создаём итоговый вектор
        b = np.zeros(M, dtype=np.float64)
        
        # Копируем свою часть
        b[0:local_M] = b_part
        
        # Получаем части от других процессов
        for i in range(1, size):
            start_row = i * local_M
            end_row = start_row + local_M
            temp = np.zeros(local_M, dtype=np.float64)
            comm.Recv(temp, source=i, tag=i)
            b[start_row:end_row] = temp
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Сохранение результата
        np.savetxt('Results_parallel_sendrecv.dat', b, fmt='%.6f')
        
        print(f"Время выполнения: {execution_time:.6f} секунд")
        print(f"Результат сохранён в Results_parallel_sendrecv.dat")
        
        return b, execution_time
    else:
        # Отправка своей части процессу 0
        comm.Send(b_part, dest=0, tag=rank)
        return None, None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 50)
        print("ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (Send/Recv)")
        print("=" * 50)
    
    result, exec_time = parallel_matrix_vector_multiply_send_recv()
    
    if rank == 0:
        print(f"\nПервые 10 элементов результата:")
        print(result[:10])
