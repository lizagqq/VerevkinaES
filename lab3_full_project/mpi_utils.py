import numpy as np
from mpi4py import MPI

def calculate_distribution(total_size, num_processes):
    """
    Вычисляет распределение данных между процессами
    
    Returns:
    rcounts - количество элементов для каждого процесса
    displs - смещения для каждого процесса
    """
    base_count = total_size // num_processes
    remainder = total_size % num_processes
    
    rcounts = []
    displs = []
    offset = 0
    
    for i in range(num_processes):
        count = base_count + (1 if i < remainder else 0)
        rcounts.append(count)
        displs.append(offset)
        offset += count
    
    return rcounts, displs

def get_local_size(rank, total_size, num_processes):
    """
    Вычисляет локальный размер данных для данного процесса
    """
    base_count = total_size // num_processes
    remainder = total_size % num_processes
    return base_count + (1 if rank < remainder else 0)

def parallel_dot_product(comm, a, b):
    """
    Параллельное вычисление скалярного произведения
    """
    local_dot = np.dot(a, b)
    global_dot = np.zeros(1, dtype=np.float64)
    comm.Allreduce(np.array([local_dot], dtype=np.float64), 
                   global_dot, op=MPI.SUM)
    return global_dot[0]

def parallel_norm(comm, vec):
    """
    Параллельное вычисление нормы вектора
    """
    return np.sqrt(parallel_dot_product(comm, vec, vec))

def print_convergence_header():
    """
    Выводит заголовок таблицы сходимости
    """
    print(f"\n{'Итерация':<10} {'||r||':<15} {'alpha':<15} {'beta':<15}")
    print("-" * 60)

def print_convergence_step(iteration, residual_norm, alpha=None, beta=None):
    """
    Выводит информацию об одной итерации
    """
    alpha_str = f"{alpha:.6e}" if alpha is not None else "N/A"
    beta_str = f"{beta:.6e}" if beta is not None else "N/A"
    print(f"{iteration:<10} {residual_norm:<15.6e} {alpha_str:<15} {beta_str:<15}")

def compare_solutions(x_computed, x_true, label=""):
    """
    Сравнивает вычисленное решение с истинным
    """
    abs_error = np.linalg.norm(x_computed - x_true)
    rel_error = abs_error / np.linalg.norm(x_true) if np.linalg.norm(x_true) > 0 else abs_error
    
    print(f"\n{label}:")
    print(f"  Абсолютная ошибка: {abs_error:.6e}")
    print(f"  Относительная ошибка: {rel_error:.6e}")
    print(f"  Норма решения: {np.linalg.norm(x_computed):.6f}")
    print(f"  Норма истинного решения: {np.linalg.norm(x_true):.6f}")
    
    return abs_error, rel_error
