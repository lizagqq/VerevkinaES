from mpi4py import MPI
import numpy as np
import time
from mpi_utils import *

def parallel_conjugate_gradient_full(
    comm, rank, size, 
    A_part, b_part, 
    M, N,
    max_iterations=None,
    tolerance=1e-10,
    verbose=False,
    alpha_reg=0.0
):
    """
    Полная параллельная реализация метода сопряжённых градиентов
    
    В этой версии векторы x, r, p распределены между процессами.
    Требуется Allgatherv для сбора полного x перед умножением на матрицу.
    
    Parameters:
    comm - MPI коммуникатор
    rank - ранг процесса
    size - количество процессов
    A_part - локальная часть матрицы (local_M x N)
    b_part - локальная часть вектора b
    M, N - размеры полной матрицы
    max_iterations - максимальное количество итераций (по умолчанию N)
    tolerance - критерий остановки по норме невязки
    verbose - выводить информацию о сходимости
    alpha_reg - параметр регуляризации Тихонова
    """
    
    if max_iterations is None:
        max_iterations = N
    
    local_M = len(b_part)
    
    # Вычисляем распределение для векторов
    rcounts_M, displs_M = calculate_distribution(M, size)
    rcounts_N, displs_N = calculate_distribution(N, size)
    
    local_N = rcounts_N[rank]
    
    # Инициализация: x = 0
    x_part = np.zeros(local_N, dtype=np.float64)
    x_full = np.zeros(N, dtype=np.float64)
    
    # Собираем полный x для первой итерации (хотя он нулевой)
    comm.Allgatherv(x_part, [x_full, rcounts_N, displs_N, MPI.DOUBLE])
    
    # Вычисляем начальную невязку: r = b - A @ x (r = b, так как x = 0)
    # q = A @ x (будет 0)
    q_part = np.zeros(local_M, dtype=np.float64)
    
    # r = b - q = b
    r_part = b_part.copy()
    
    # p = A.T @ r (направление спуска)
    p_part = np.dot(A_part.T, r_part)
    
    # Собираем полный p с помощью Allreduce(SUM)
    p_full = np.zeros(N, dtype=np.float64)
    comm.Allreduce(p_part, p_full, op=MPI.SUM)
    
    # Распределяем p обратно по процессам
    p_part = p_full[displs_N[rank]:displs_N[rank] + local_N].copy()
    
    # gamma_old = ||r||^2
    gamma_old = parallel_dot_product(comm, r_part, r_part)
    
    if verbose and rank == 0:
        print_convergence_header()
        print_convergence_step(0, np.sqrt(gamma_old))
    
    # Основной цикл
    for iteration in range(1, max_iterations + 1):
        # Собираем полный p для умножения на матрицу
        p_full = np.zeros(N, dtype=np.float64)
        comm.Allgatherv(p_part, [p_full, rcounts_N, displs_N, MPI.DOUBLE])
        
        # q = A @ p (локально)
        q_part = np.dot(A_part, p_full)
        
        # Добавляем регуляризацию: q += alpha * p
        if alpha_reg > 0:
            q_part += alpha_reg * p_part
        
        # delta = p^T @ q (скалярное произведение)
        delta = parallel_dot_product(comm, p_part, q_part)
        
        # Избегаем деления на ноль
        if abs(delta) < 1e-16:
            if rank == 0 and verbose:
                print(f"Остановка: delta = {delta:.2e} (слишком мало)")
            break
        
        # alpha = gamma_old / delta
        alpha = gamma_old / delta
        
        # Обновление решения: x = x + alpha * p
        x_part += alpha * p_part
        
        # Обновление невязки: r = r - alpha * q
        r_part -= alpha * q_part
        
        # gamma_new = ||r||^2
        gamma_new = parallel_dot_product(comm, r_part, r_part)
        
        residual_norm = np.sqrt(gamma_new)
        
        # Критерий остановки по норме невязки
        if residual_norm < tolerance:
            if rank == 0 and verbose:
                print_convergence_step(iteration, residual_norm, alpha)
                print(f"\nСходимость достигнута: ||r|| = {residual_norm:.2e} < {tolerance:.2e}")
            break
        
        # beta = gamma_new / gamma_old
        beta = gamma_new / gamma_old
        
        # Обновление направления: p = A.T @ r + beta * p
        # Сначала вычисляем A.T @ r
        temp_part = np.dot(A_part.T, r_part)
        
        # Собираем с помощью Allreduce
        temp_full = np.zeros(N, dtype=np.float64)
        comm.Allreduce(temp_part, temp_full, op=MPI.SUM)
        
        # Распределяем обратно
        temp_part = temp_full[displs_N[rank]:displs_N[rank] + local_N].copy()
        
        # p = temp + beta * p
        p_part = temp_part + beta * p_part
        
        gamma_old = gamma_new
        
        if verbose and rank == 0 and iteration % 10 == 0:
            print_convergence_step(iteration, residual_norm, alpha, beta)
    
    # Собираем финальное решение на процессе 0
    x_final = None
    if rank == 0:
        x_final = np.zeros(N, dtype=np.float64)
    
    comm.Gatherv(x_part, [x_final, rcounts_N, displs_N, MPI.DOUBLE], root=0)
    
    return x_final, iteration, residual_norm

def main():
    """
    Основная функция для запуска полной версии CG
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
        print("ПОЛНАЯ ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ МЕТОДА СОПРЯЖЁННЫХ ГРАДИЕНТОВ")
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
    
    # Распределение данных
    A_part = np.zeros((local_M, N), dtype=np.float64)
    b_part = np.zeros(local_M, dtype=np.float64)
    
    sendcounts_A = [rc * N for rc in rcounts_M]
    displs_A = [d * N for d in displs_M]
    
    comm.Scatterv([A, sendcounts_A, displs_A, MPI.DOUBLE], A_part, root=0)
    comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE], b_part, root=0)
    
    # Запуск метода
    if rank == 0:
        print("\nЗапуск метода сопряжённых градиентов...")
    
    start_time = time.time()
    
    x_solution, iterations, final_residual = parallel_conjugate_gradient_full(
        comm, rank, size,
        A_part, b_part,
        M, N,
        verbose=(rank == 0),
        tolerance=1e-10
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Результаты и верификация на процессе 0
    if rank == 0:
        print(f"\n{'='*70}")
        print("РЕЗУЛЬТАТЫ")
        print(f"{'='*70}")
        print(f"Количество итераций: {iterations}")
        print(f"Финальная норма невязки: {final_residual:.6e}")
        print(f"Время выполнения: {execution_time:.6f} сек")
        
        # Верификация с numpy
        print("\nВерификация с numpy.linalg.lstsq...")
        x_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
        
        compare_solutions(x_solution, x_numpy, "Сравнение с numpy.linalg.lstsq")
        
        # Сравнение с истинным решением, если доступно
        if x_true is not None:
            compare_solutions(x_solution, x_true, "Сравнение с истинным решением")
        
        # Проверка невязки
        residual = b - A @ x_solution
        residual_norm = np.linalg.norm(residual)
        print(f"\nНорма невязки (||b - Ax||): {residual_norm:.6e}")
        
        # Сохранение решения
        np.savetxt('x_solution_full.dat', x_solution, fmt='%.10f')
        print(f"\nРешение сохранено в x_solution_full.dat")
        
        return execution_time
    
    return execution_time

if __name__ == "__main__":
    main()
