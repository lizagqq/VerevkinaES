from mpi4py import MPI
import numpy as np
import time
import sys

def parallel_dot_product(use_allreduce=False):
    """
    Параллельное вычисление скалярного произведения
    
    Parameters:
    use_allreduce - если True, использует Allreduce, иначе Reduce
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    a = None
    M = None
    
    if rank == 0:
        a = np.loadtxt('vector_a.dat', dtype=np.float64)
        M = len(a)
        print(f"\nДлина вектора: {M}")
        print(f"Количество процессов: {size}")
    
    M = comm.bcast(M, root=0)
    
    # Вычисление распределения
    base_local_M = M // size
    remainder = M % size
    
    if rank < remainder:
        local_M = base_local_M + 1
    else:
        local_M = base_local_M
    
    a_part = np.zeros(local_M, dtype=np.float64)
    
    if rank == 0:
        sendcounts = []
        displacements = []
        offset = 0
        
        for i in range(size):
            count = base_local_M + 1 if i < remainder else base_local_M
            sendcounts.append(count)
            displacements.append(offset)
            offset += count
        
        print(f"Распределение элементов: {sendcounts}")
    else:
        sendcounts = None
        displacements = None
    
    start_time = time.time()
    
    comm.Scatterv([a, sendcounts, displacements, MPI.DOUBLE], a_part, root=0)
    local_dot = np.dot(a_part, a_part)
    
    if use_allreduce:
        global_dot = np.zeros(1, dtype=np.float64)
        comm.Allreduce(np.array([local_dot], dtype=np.float64), global_dot, op=MPI.SUM)
    else:
        global_dot = np.zeros(1, dtype=np.float64)
        comm.Reduce(np.array([local_dot], dtype=np.float64), global_dot, op=MPI.SUM, root=0)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    if rank == 0:
        seq_start = time.time()
        seq_dot = np.dot(a, a)
        seq_end = time.time()
        seq_time = seq_end - seq_start
        
        operation_name = "Allreduce" if use_allreduce else "Reduce"
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ (MPI.{operation_name})")
        print(f"{'='*60}")
        print(f"Параллельный результат: {global_dot[0]:.10f}")
        print(f"Последовательный результат: {seq_dot:.10f}")
        print(f"Абсолютная ошибка: {abs(global_dot[0] - seq_dot):.2e}")
        print(f"Относительная ошибка: {abs(global_dot[0] - seq_dot) / abs(seq_dot):.2e}")
        print(f"Время параллельное: {execution_time:.6f} сек")
        print(f"Время последовательное: {seq_time:.6f} сек")
        print(f"Ускорение: {seq_time / execution_time:.2f}x")
        
        return global_dot[0], execution_time, seq_time
    
    return global_dot[0] if use_allreduce else None, execution_time, None

def parallel_transpose_matrix_vector():
    """
    Параллельное вычисление b = A.T @ x
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    M = None
    N = None
    A = None
    x = None
    
    if rank == 0:
        with open('in.dat', 'r') as f:
            M, N = map(int, f.readline().split())
        
        print(f"\nРазмер матрицы: {M} x {N}")
        print(f"Количество процессов: {size}")
        
        A = np.loadtxt('AData.dat', dtype=np.float64).reshape(M, N)
        x = np.loadtxt('xData.dat', dtype=np.float64)
    
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    
    base_local_M = M // size
    remainder = M % size
    
    if rank < remainder:
        local_M = base_local_M + 1
    else:
        local_M = base_local_M
    
    A_part = np.zeros((local_M, N), dtype=np.float64)
    x_part = np.zeros(local_M, dtype=np.float64)
    
    if rank == 0:
        sendcounts_A = []
        displacements_A = []
        sendcounts_x = []
        displacements_x = []
        offset_A = 0
        offset_x = 0
        
        for i in range(size):
            count_rows = base_local_M + 1 if i < remainder else base_local_M
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
    
    comm.Scatterv([A, sendcounts_A, displacements_A, MPI.DOUBLE], A_part, root=0)
    comm.Scatterv([x, sendcounts_x, displacements_x, MPI.DOUBLE], x_part, root=0)
    
    b_temp = np.dot(A_part.T, x_part)
    
    b = np.zeros(N, dtype=np.float64) if rank == 0 else None
    comm.Reduce(b_temp, b, op=MPI.SUM, root=0)
    
    end_time = time.time()
    
    if rank == 0:
        execution_time = end_time - start_time
        
        seq_start = time.time()
        b_seq = np.dot(A.T, x)
        seq_end = time.time()
        seq_time = seq_end - seq_start
        
        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТЫ")
        print(f"{'='*60}")
        
        abs_error = np.max(np.abs(b - b_seq))
        rel_error = abs_error / np.max(np.abs(b_seq)) if np.max(np.abs(b_seq)) > 0 else 0
        
        print(f"Макс. абсолютная ошибка: {abs_error:.2e}")
        print(f"Макс. относительная ошибка: {rel_error:.2e}")
        print(f"Первые 10 элементов (параллельный): {b[:10]}")
        print(f"Первые 10 элементов (последовательный): {b_seq[:10]}")
        print(f"Время параллельное: {execution_time:.6f} сек")
        print(f"Время последовательное: {seq_time:.6f} сек")
        print(f"Ускорение: {seq_time / execution_time:.2f}x")
        
        np.savetxt('Results_part2_parallel.dat', b, fmt='%.10f')
        
        return b, execution_time, seq_time
    
    return None, None, None

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Определяем, какую часть запускать
    if len(sys.argv) > 1:
        part = sys.argv[1]
    else:
        part = "both"
    
    if rank == 0:
        print("=" * 60)
        print("ЛАБОРАТОРНАЯ РАБОТА №2")
        print("=" * 60)
    
    if part in ["1", "part1", "both"]:
        if rank == 0:
            print("\n" + "=" * 60)
            print("ЧАСТЬ 1: СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ")
            print("=" * 60)
        
        # Вариант А: Reduce
        parallel_dot_product(use_allreduce=False)
        
        # Вариант Б: Allreduce
        parallel_dot_product(use_allreduce=True)
    
    if part in ["2", "part2", "both"]:
        if rank == 0:
            print("\n" + "=" * 60)
            print("ЧАСТЬ 2: УМНОЖЕНИЕ A.T @ x")
            print("=" * 60)
        
        parallel_transpose_matrix_vector()
