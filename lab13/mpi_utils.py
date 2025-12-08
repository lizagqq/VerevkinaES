import numpy as np
from mpi4py import MPI

def get_local_size(rank, total_size, num_procs):
    """
    Вычисляет размер локальной части данных для процесса
    """
    base_size = total_size // num_procs
    remainder = total_size % num_procs
    
    if rank < remainder:
        return base_size + 1
    else:
        return base_size

def calculate_distribution(total_size, num_procs):
    """
    Вычисляет распределение данных между процессами
    Возвращает массивы размеров и смещений для Scatterv/Gatherv
    """
    rcounts = np.array([get_local_size(i, total_size, num_procs) 
                        for i in range(num_procs)], dtype=np.int32)
    displs = np.zeros(num_procs, dtype=np.int32)
    displs[1:] = np.cumsum(rcounts[:-1])
    
    return rcounts, displs

def parallel_dot_product(comm, local_x, local_y):
    """
    Параллельное вычисление скалярного произведения
    """
    local_result = np.dot(local_x, local_y)
    global_result = comm.allreduce(local_result, op=MPI.SUM)
    return global_result

def print_convergence_header():
    """
    Печать заголовка таблицы сходимости
    """
    print(f"\n{'Итерация':<12} {'||r||':<15} {'alpha':<15} {'beta':<15}")
    print("-" * 60)

def print_convergence_step(iteration, residual_norm, alpha=None, beta=None):
    """
    Печать информации об итерации
    """
    alpha_str = f"{alpha:.6e}" if alpha is not None else "-"
    beta_str = f"{beta:.6e}" if beta is not None else "-"
    print(f"{iteration:<12} {residual_norm:<15.6e} {alpha_str:<15} {beta_str:<15}")

def compare_solutions(x1, x2, title="Сравнение решений"):
    """
    Сравнивает два решения и выводит метрики
    """
    abs_error = np.linalg.norm(x1 - x2)
    rel_error = abs_error / (np.linalg.norm(x2) + 1e-16)
    
    print(f"\n{title}:")
    print(f"  Абсолютная ошибка: {abs_error:.6e}")
    print(f"  Относительная ошибка: {rel_error:.6e}")
    
    return abs_error, rel_error
