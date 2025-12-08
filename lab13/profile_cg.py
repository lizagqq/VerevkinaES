from mpi4py import MPI
import numpy as np
import time
import cProfile
import pstats
import io
from mpi_utils import *

def profile_cg_simple(A_part, b, M, N, num_iterations=10):
    """
    Профилирование упрощённой версии CG
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Импортируем функцию CG
    from cg_simple import parallel_conjugate_gradient_simple
    
    profiler = cProfile.Profile()
    
    # Профилирование
    profiler.enable()
    x_solution, iterations, residual = parallel_conjugate_gradient_simple(
        comm, rank, size,
        A_part, b,
        M, N,
        max_iterations=num_iterations,
        verbose=False
    )
    profiler.disable()
    
    return profiler, x_solution, iterations, residual

def analyze_profile(profiler, rank, output_file=None):
    """
    Анализ результатов профилирования
    """
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    
    if rank == 0:
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТЫ ПРОФИЛИРОВАНИЯ")
        print("="*70)
        
        # Топ функций по совокупному времени
        print("\nТоп-20 функций по совокупному времени:")
        ps.print_stats(20)
        
        if output_file:
            with open(output_file, 'w') as f:
                ps = pstats.Stats(profiler, stream=f)
                ps.sort_stats('cumulative')
                ps.print_stats()
            print(f"\nПолный отчёт сохранён в {output_file}")

def measure_communication_overhead():
    """
    Измерение накладных расходов на коммуникацию
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Размеры для тестирования
    sizes = [100, 1000, 10000, 100000]
    
    results = {
        'allreduce': [],
        'allgatherv': [],
        'scatterv': []
    }
    
    for n in sizes:
        data = np.random.randn(n)
        
        # Тест Allreduce
        comm.Barrier()
        start = time.time()
        for _ in range(100):
            result = comm.allreduce(np.sum(data), op=MPI.SUM)
        comm.Barrier()
        results['allreduce'].append((time.time() - start) / 100)
        
        # Тест Allgatherv
        local_size = n // size
        local_data = np.random.randn(local_size)
        full_data = np.zeros(n)
        rcounts, displs = calculate_distribution(n, size)
        
        comm.Barrier()
        start = time.time()
        for _ in range(100):
            comm.Allgatherv(local_data[:rcounts[rank]], 
                           [full_data, rcounts, displs, MPI.DOUBLE])
        comm.Barrier()
        results['allgatherv'].append((time.time() - start) / 100)
        
        # Тест Scatterv
        if rank == 0:
            send_data = np.random.randn(n)
        else:
            send_data = None
        recv_data = np.zeros(local_size)
        
        comm.Barrier()
        start = time.time()
        for _ in range(100):
            if rank == 0:
                comm.Scatterv([send_data, rcounts, displs, MPI.DOUBLE], 
                             recv_data[:rcounts[rank]], root=0)
            else:
                comm.Scatterv([None, rcounts, displs, MPI.DOUBLE], 
                             recv_data[:rcounts[rank]], root=0)
        comm.Barrier()
        results['scatterv'].append((time.time() - start) / 100)
    
    if rank == 0:
        print("\n" + "="*70)
        print("НАКЛАДНЫЕ РАСХОДЫ НА КОММУНИКАЦИЮ")
        print("="*70)
        print(f"\n{'Размер':<12} {'Allreduce':<15} {'Allgatherv':<15} {'Scatterv':<15}")
        print("-" * 70)
        for i, n in enumerate(sizes):
            print(f"{n:<12} {results['allreduce'][i]*1000:<15.6f} "
                  f"{results['allgatherv'][i]*1000:<15.6f} "
                  f"{results['scatterv'][i]*1000:<15.6f}")
        print("\n(время в миллисекундах)")
    
    return results

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print("ПРОФИЛИРОВАНИЕ МЕТОДА СОПРЯЖЁННЫХ ГРАДИЕНТОВ")
        print("="*70)
    
    # Загрузка данных
    M, N, A, b = None, None, None, None
    
    if rank == 0:
        with open('in.dat', 'r') as f:
            M, N = map(int, f.readline().split())
        A = np.loadtxt('AData.dat', dtype=np.float64).reshape(M, N)
        b = np.loadtxt('bData.dat', dtype=np.float64)
        print(f"\nРазмер системы: {M} x {N}")
        print(f"Количество процессов: {size}")
    
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    
    # Распределение данных
    local_M = get_local_size(rank, M, size)
    rcounts_M, displs_M = calculate_distribution(M, size)
    
    A_part = np.zeros((local_M, N), dtype=np.float64)
    sendcounts_A = [rc * N for rc in rcounts_M]
    displs_A = [d * N for d in displs_M]
    
    comm.Scatterv([A, sendcounts_A, displs_A, MPI.DOUBLE], A_part, root=0)
    b_full = comm.bcast(b, root=0)
    
    # Профилирование
    if rank == 0:
        print("\nЗапуск профилирования...")
    
    profiler, x_solution, iterations, residual = profile_cg_simple(
        A_part, b_full, M, N, num_iterations=50
    )
    
    # Анализ профилирования
    analyze_profile(profiler, rank, 
                   f'profile_rank_{rank}.txt' if rank == 0 else None)
    
    # Измерение накладных расходов на коммуникацию
    if rank == 0:
        print("\n\nИзмерение накладных расходов на коммуникацию...")
    
    comm_results = measure_communication_overhead()
    
    if rank == 0:
        print("\nПрофилирование завершено")

if __name__ == "__main__":
    main()
