from mpi4py import MPI
import numpy as np
import time
from mpi_utils import *

def parallel_conjugate_gradient_simple(
    comm, rank, size,
    A_part, b,
    M, N,
    max_iterations=None,
    tolerance=1e-10,
    verbose=False,
    alpha_reg=0.0
):
    """
    Упрощённая параллельная реализация метода сопряжённых градиентов
    Решает нормальные уравнения: (A^T A)x = A^T b
    
    В этой версии полные векторы x, r, p хранятся на каждом процессе.
    Параллельными являются только операции с матрицей.
    
    Parameters:
    comm - MPI коммуникатор
    rank - ранг процесса  
    size - количество процессов
    A_part - локальная часть матрицы (local_M x N)
    b - полный вектор b (доступен на всех процессах)
    M, N - размеры полной матрицы
    max_iterations - максимальное количество итераций
    tolerance - критерий остановки по норме невязки
    verbose - выводить информацию о сходимости
    alpha_reg - параметр регуляризации Тихонова
    """
    
    if max_iterations is None:
        max_iterations = N
    
    local_M = A_part.shape[0]
    rcounts, displs = calculate_distribution(M, size)
    
    # Инициализация: x = 0
    x = np.zeros(N, dtype=np.float64)
    
    # Вычисляем правую часть нормальных уравнений: c = A^T b
    b_part = b[displs[rank]:displs[rank] + local_M]
    c_part = np.dot(A_part.T, b_part)
    c = np.zeros(N, dtype=np.float64)
    comm.Allreduce(c_part, c, op=MPI.SUM)
    
    # Начальная невязка: r = c - (A^T A) x = c (так как x = 0)
    r = c.copy()
    
    # Начальное направление: p = r
    p = r.copy()
    
    # gamma_old = r^T @ r
    gamma_old = np.dot(r, r)
    
    if verbose and rank == 0:
        print_convergence_header()
        print_convergence_step(0, np.sqrt(gamma_old))
    
    # Основной цикл
    for iteration in range(1, max_iterations + 1):
        # Вычисляем s = (A^T A) @ p
        # Шаг 1: q = A @ p (каждый процесс вычисляет свою часть)
        q_part = np.dot(A_part, p)
        
        # Шаг 2: s = A^T @ q (собираем с помощью Allreduce)
        s_part = np.dot(A_part.T, q_part)
        s = np.zeros(N, dtype=np.float64)
        comm.Allreduce(s_part, s, op=MPI.SUM)
        
        # Добавляем регуляризацию: s += alpha * p
        if alpha_reg > 0:
            s += alpha_reg * p
        
        # delta = p^T @ s
        delta = np.dot(p, s)
        
        # Избегаем деления на ноль
        if abs(delta) < 1e-16:
            if rank == 0 and verbose:
                print(f"Остановка: delta = {delta:.2e} (слишком мало)")
            break
        
        # alpha = gamma_old / delta
        alpha = gamma_old / delta
        
        # Обновление решения: x = x + alpha * p
        x += alpha * p
        
        # Обновление невязки: r = r - alpha * s
        r -= alpha * s
        
        # gamma_new = r^T @ r
        gamma_new = np.dot(r, r)
        
        residual_norm = np.sqrt(gamma_new)
        
        # Критерий остановки по норме невязки
        if residual_norm < tolerance:
            if rank == 0 and verbose:
                print_convergence_step(iteration, residual_norm, alpha)
                print(f"\nСходимость достигнута: ||r|| = {residual_norm:.2e} < {tolerance:.2e}")
            break
        
        # beta = gamma_new / gamma_old
        beta = gamma_new / gamma_old
        
        # Обновление направления: p = r + beta * p
        p = r + beta * p
        
        gamma_old = gamma_new
        
        if verbose and rank == 0 and iteration % 10 == 0:
            print_convergence_step(iteration, residual_norm, alpha, beta)
    
    return x, iteration, residual_norm

def main():
    """
    Основная функция для запуска упрощённой версии CG
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Чтение параметров
    M = None
    N = None
    A = None
    b = None
    x_true = None
    
    if rank == 0:
        print("=" * 70)
        print("УПРОЩЁННАЯ ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ МЕТОДА СОПРЯЖЁННЫХ ГРАДИЕНТОВ")
        print("=" * 70)
        
        # Чтение размеров
        with open('in.dat', 'r') as f:
            M, N = map(int, f.readline().split())
        
        print(f"\nРазмер системы: {M} x {N}")
        print(f"Количество процессов: {size}")
        
        # Чтение данных
        A = np.loadtxt('AData.dat', dtype=np.float64).reshape(M, N)
        b = np.loadtxt('bData.dat', dtype=np.float64)
        
        try:
            x_true = np.loadtxt('x_true.dat', dtype=np.float64)
        except:
            x_true = None
    
    # Рассылка размеров
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    
    # Вычисление локальных размеров
    local_M = get_local_size(rank, M, size)
    rcounts_M, displs_M = calculate_distribution(M, size)
    
    # Распределение матрицы A
    A_part = np.zeros((local_M, N), dtype=np.float64)
    
    sendcounts_A = [rc * N for rc in rcounts_M]
    displs_A = [d * N for d in displs_M]
    
    comm.Scatterv([A, sendcounts_A, displs_A, MPI.DOUBLE], A_part, root=0)
    
    # Рассылка полного вектора b всем процессам
    b_full = comm.bcast(b, root=0)
    
    # Запуск метода
    if rank == 0:
        print("\nЗапуск метода сопряжённых градиентов...")
    
    start_time = time.time()
    
    x_solution, iterations, final_residual = parallel_conjugate_gradient_simple(
        comm, rank, size,
        A_part, b_full,
        M, N,
        verbose=(rank == 0),
        tolerance=1e-10
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Результаты и верификация на всех процессах (но выводим только на 0)
    if rank == 0:
        print(f"\n{'='*70}")
        print("РЕЗУЛЬТАТЫ")
        print(f"{'='*70}")
        print(f"Количество итераций: {iterations}")
        print(f"Финальная норма невязки (нормальных уравнений): {final_residual:.6e}")
        print(f"Время выполнения: {execution_time:.6f} сек")
        
        # Верификация с numpy
        print("\nВерификация с numpy.linalg.lstsq...")
        x_numpy = np.linalg.lstsq(A, b_full, rcond=None)[0]
        
        compare_solutions(x_solution, x_numpy, "Сравнение с numpy.linalg.lstsq")
        
        # Сравнение с истинным решением, если доступно
        if x_true is not None:
            compare_solutions(x_solution, x_true, "Сравнение с истинным решением")
        
        # Проверка невязки исходной системы Ax = b
        residual = b_full - A @ x_solution
        residual_norm = np.linalg.norm(residual)
        print(f"\nНорма невязки исходной системы (||b - Ax||): {residual_norm:.6e}")
        
        # Сохранение решения
        np.savetxt('x_solution_simple.dat', x_solution, fmt='%.10f')
        print(f"\nРешение сохранено в x_solution_simple.dat")
        
        return execution_time
    
    return execution_time

if __name__ == "__main__":
    main()
