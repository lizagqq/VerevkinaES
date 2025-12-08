from mpi4py import MPI
import numpy as np
import time
from mpi_utils import *

def parallel_cg_regularized(
    comm, rank, size,
    A_part, b,
    M, N,
    alpha_reg=0.01,
    max_iterations=None,
    tolerance=1e-10,
    early_stop=True,
    verbose=False
):
    """
    Метод сопряжённых градиентов с регуляризацией Тихонова
    
    Решает: (A^T A + alpha * I) x = A^T b
    что эквивалентно минимизации: ||Ax - b||^2 + alpha * ||x||^2
    
    Parameters:
    alpha_reg - параметр регуляризации Тихонова
    early_stop - использовать ранний останов по норме невязки
    """
    
    if max_iterations is None:
        max_iterations = N
    
    local_M = A_part.shape[0]
    rcounts, displs = calculate_distribution(M, size)
    
    # Инициализация: x = 0
    x = np.zeros(N, dtype=np.float64)
    
    # Вычисляем r = b
    r = b.copy()
    
    # p = A.T @ r
    r_part = r[displs[rank]:displs[rank] + local_M]
    p_part = np.dot(A_part.T, r_part)
    p = np.zeros(N, dtype=np.float64)
    comm.Allreduce(p_part, p, op=MPI.SUM)
    
    # gamma_old = r^T @ r
    gamma_old = np.dot(r, r)
    initial_residual = np.sqrt(gamma_old)
    
    if verbose and rank == 0:
        print(f"\nПараметр регуляризации: alpha = {alpha_reg:.2e}")
        print(f"Ранний останов: {'Да' if early_stop else 'Нет'}")
        print(f"Начальная норма невязки: {initial_residual:.6e}")
        print_convergence_header()
        print_convergence_step(0, initial_residual)
    
    iteration_history = []
    residual_history = []
    
    # Основной цикл
    for iteration in range(1, max_iterations + 1):
        # q = A @ p + alpha * p (регуляризация)
        q_part = np.dot(A_part, p)
        q = np.zeros(M, dtype=np.float64)
        comm.Allgatherv(q_part, [q, rcounts, displs, MPI.DOUBLE])
        
        # Добавляем регуляризацию
        q += alpha_reg * p
        
        # delta = p^T @ q
        delta = np.dot(p, q)
        
        if abs(delta) < 1e-16:
            if rank == 0 and verbose:
                print(f"Остановка: delta = {delta:.2e}")
            break
        
        # alpha = gamma_old / delta
        alpha = gamma_old / delta
        
        # x = x + alpha * p
        x += alpha * p
        
        # r = r - alpha * q
        r -= alpha * q
        
        # gamma_new = r^T @ r
        gamma_new = np.dot(r, r)
        residual_norm = np.sqrt(gamma_new)
        
        iteration_history.append(iteration)
        residual_history.append(residual_norm)
        
        # Ранний останов по норме невязки
        if early_stop and residual_norm < tolerance:
            if rank == 0 and verbose:
                print_convergence_step(iteration, residual_norm, alpha)
                print(f"\nРанний останов: ||r|| = {residual_norm:.2e} < {tolerance:.2e}")
            break
        
        # beta = gamma_new / gamma_old
        beta = gamma_new / gamma_old
        
        # p = A.T @ r + beta * p
        r_part = r[displs[rank]:displs[rank] + local_M]
        temp_part = np.dot(A_part.T, r_part)
        temp = np.zeros(N, dtype=np.float64)
        comm.Allreduce(temp_part, temp, op=MPI.SUM)
        
        p = temp + beta * p
        
        gamma_old = gamma_new
        
        if verbose and rank == 0 and (iteration % 10 == 0 or iteration == max_iterations):
            print_convergence_step(iteration, residual_norm, alpha, beta)
    
    return x, iteration, residual_norm, iteration_history, residual_history

def test_regularization_parameter():
    """
    Тестирование различных значений параметра регуляризации
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Загрузка данных
    M, N, A, b, x_true = None, None, None, None, None
    
    if rank == 0:
        print("=" * 70)
        print("ТЕСТИРОВАНИЕ РЕГУЛЯРИЗАЦИИ ТИХОНОВА")
        print("=" * 70)
        
        with open('ill_in.dat', 'r') as f:
            M, N = map(int, f.readline().split())
        
        A = np.loadtxt('ill_AData.dat', dtype=np.float64).reshape(M, N)
        b = np.loadtxt('ill_bData.dat', dtype=np.float64)
        
        try:
            x_true = np.loadtxt('ill_x_true.dat', dtype=np.float64)
        except:
            x_true = None
        
        print(f"\nРазмер системы: {M} x {N}")
        print(f"Число обусловленности: {np.linalg.cond(A):.2e}")
    
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
    
    # Тестируем разные значения alpha
    alphas = [0.0, 0.001, 0.01, 0.1, 1.0]
    results = {}
    
    for alpha_val in alphas:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Тестирование alpha = {alpha_val:.3f}")
            print(f"{'='*70}")
        
        x_sol, iters, res_norm, _, _ = parallel_cg_regularized(
            comm, rank, size,
            A_part, b_full,
            M, N,
            alpha_reg=alpha_val,
            tolerance=1e-10,
            verbose=(rank == 0)
        )
        
        if rank == 0:
            # Сравнение с истинным решением
            if x_true is not None:
                abs_err, rel_err = compare_solutions(x_sol, x_true, 
                                                     f"Alpha = {alpha_val:.3f}")
                results[alpha_val] = {
                    'iterations': iters,
                    'residual': res_norm,
                    'abs_error': abs_err,
                    'rel_error': rel_err
                }
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print(f"{'='*70}")
        print(f"{'Alpha':<10} {'Итерации':<12} {'Невязка':<15} {'Отн. ошибка':<15}")
        print("-" * 70)
        for alpha_val in alphas:
            if alpha_val in results:
                r = results[alpha_val]
                print(f"{alpha_val:<10.3f} {r['iterations']:<12} "
                      f"{r['residual']:<15.2e} {r['rel_error']:<15.2e}")

def main():
    """
    Основная функция
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 70)
        print("МЕТОД СОПРЯЖЁННЫХ ГРАДИЕНТОВ С РЕГУЛЯРИЗАЦИЕЙ")
        print("=" * 70)
        print("\n1. Тестирование на хорошо обусловленной системе")
    
    # Здесь можно добавить код для основного теста
    
    # Запуск теста регуляризации
    test_regularization_parameter()

if __name__ == "__main__":
    main()
