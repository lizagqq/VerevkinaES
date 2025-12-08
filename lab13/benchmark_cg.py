from mpi4py import MPI
import numpy as np
import time
import json
from mpi_utils import *

def benchmark_version(comm, rank, size, A_part, b, M, N, version_func, version_name, num_runs=5):
    """
    Бенчмарк одной версии алгоритма
    """
    times = []
    iterations_list = []
    residuals = []
    
    for run in range(num_runs):
        comm.Barrier()
        start = time.time()
        
        x, iters, res = version_func(
            comm, rank, size,
            A_part, b, M, N,
            verbose=False,
            tolerance=1e-10
        )
        
        comm.Barrier()
        elapsed = time.time() - start
        
        times.append(elapsed)
        iterations_list.append(iters)
        residuals.append(res)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return {
        'version': version_name,
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'avg_iterations': np.mean(iterations_list),
        'avg_residual': np.mean(residuals),
        'all_times': times
    }

def benchmark_scalability(problem_sizes, num_procs_list):
    """
    Тестирование масштабируемости
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size not in num_procs_list:
        if rank == 0:
            print(f"Внимание: текущее количество процессов {size} не в списке тестирования")
        return None
    
    results = {}
    
    # Импортируем версии
    from cg_simple import parallel_conjugate_gradient_simple
    from cg_optimized import parallel_cg_optimized
    
    for M, N in problem_sizes:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Тестирование размера: {M} x {N}")
            print(f"{'='*70}")
        
        # Генерация данных
        if rank == 0:
            from generate_data import generate_test_data
            A, b, x_true = generate_test_data(M, N, condition_number=10)
        else:
            A, b = None, None
        
        # Распределение данных
        local_M = get_local_size(rank, M, size)
        rcounts_M, displs_M = calculate_distribution(M, size)
        
        A_part = np.zeros((local_M, N), dtype=np.float64)
        sendcounts_A = [rc * N for rc in rcounts_M]
        displs_A = [d * N for d in displs_M]
        
        comm.Scatterv([A, sendcounts_A, displs_A, MPI.DOUBLE], A_part, root=0)
        b_full = comm.bcast(b, root=0)
        
        # Бенчмарк простой версии
        if rank == 0:
            print(f"\nТестирование простой версии...")
        
        result_simple = benchmark_version(
            comm, rank, size, A_part, b_full, M, N,
            parallel_conjugate_gradient_simple,
            "simple", num_runs=3
        )
        
        # Бенчмарк оптимизированной версии (без async)
        if rank == 0:
            print(f"Тестирование оптимизированной версии (без async)...")
        
        def cg_opt_no_async(comm, rank, size, A_part, b, M, N, **kwargs):
            return parallel_cg_optimized(
                comm, rank, size, A_part, b, M, N,
                use_async=False, overlap_compute=False, **kwargs
            )
        
        result_opt_basic = benchmark_version(
            comm, rank, size, A_part, b_full, M, N,
            cg_opt_no_async,
            "optimized_basic", num_runs=3
        )
        
        # Бенчмарк оптимизированной версии (с async)
        if rank == 0:
            print(f"Тестирование оптимизированной версии (с async)...")
        
        def cg_opt_async(comm, rank, size, A_part, b, M, N, **kwargs):
            return parallel_cg_optimized(
                comm, rank, size, A_part, b, M, N,
                use_async=True, overlap_compute=True, **kwargs
            )
        
        result_opt_async = benchmark_version(
            comm, rank, size, A_part, b_full, M, N,
            cg_opt_async,
            "optimized_async", num_runs=3
        )
        
        key = f"{M}x{N}"
        results[key] = {
            'size': (M, N),
            'num_procs': size,
            'simple': result_simple,
            'optimized_basic': result_opt_basic,
            'optimized_async': result_opt_async
        }
        
        if rank == 0:
            print(f"\nРезультаты для {M}x{N}:")
            print(f"  Простая версия: {result_simple['avg_time']:.6f} ± {result_simple['std_time']:.6f} сек")
            print(f"  Оптимизированная (базовая): {result_opt_basic['avg_time']:.6f} ± {result_opt_basic['std_time']:.6f} сек")
            print(f"  Оптимизированная (async): {result_opt_async['avg_time']:.6f} ± {result_opt_async['std_time']:.6f} сек")
            
            speedup_basic = result_simple['avg_time'] / result_opt_basic['avg_time']
            speedup_async = result_simple['avg_time'] / result_opt_async['avg_time']
            print(f"  Ускорение (базовая): {speedup_basic:.3f}x")
            print(f"  Ускорение (async): {speedup_async:.3f}x")
    
    return results

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*70)
        print(f"\nКоличество процессов: {size}")
    
    # Размеры задач для тестирования
    problem_sizes = [
        (500, 250),    # Малая
        (1000, 500),   # Средняя
        (2000, 1000),  # Большая
    ]
    
    # Тестирование для текущего количества процессов
    results = benchmark_scalability(problem_sizes, [1, 2, 4, 8])
    
    # Сохранение результатов
    if rank == 0 and results:
        output_file = f'benchmark_results_p{size}.json'
        with open(output_file, 'w') as f:
            # Преобразуем numpy типы в Python типы для JSON
            json_results = {}
            for key, val in results.items():
                json_results[key] = {
                    'size': val['size'],
                    'num_procs': val['num_procs'],
                    'simple': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                              for k, v in val['simple'].items() if k != 'all_times'},
                    'optimized_basic': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                       for k, v in val['optimized_basic'].items() if k != 'all_times'},
                    'optimized_async': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                       for k, v in val['optimized_async'].items() if k != 'all_times'}
                }
            
            json.dump(json_results, f, indent=2)
        
        print(f"\n\nРезультаты сохранены в {output_file}")

if __name__ == "__main__":
    main()
