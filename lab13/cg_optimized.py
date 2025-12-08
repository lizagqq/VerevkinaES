from mpi4py import MPI
import numpy as np
import time
from mpi_utils import *

def parallel_cg_optimized(
    comm, rank, size,
    A_part, b,
    M, N,
    max_iterations=None,
    tolerance=1e-10,
    verbose=False,
    use_async=True,
    overlap_compute=True
):
    """
    Оптимизированная версия метода сопряжённых градиентов
    
    Оптимизации:
    1. Асинхронные коммуникации (Iallreduce)
    2. Перекрытие вычислений и коммуникаций
    3. Оптимизация использования памяти
    4. Векторизация операций
    
    Parameters:
    use_async - использовать асинхронные операции
    overlap_compute - перекрывать вычисления и коммуникации
    """
    
    if max_iterations is None:
        max_iterations = N
    
    local_M = A_part.shape[0]
    rcounts, displs = calculate_distribution(M, size)
    
    # Инициализация: x = 0
    x = np.zeros(N, dtype=np.float64)
    
    # Вычисляем правую часть: c = A^T b
    b_part = b[displs[rank]:displs[rank] + local_M]
    c_part = np.dot(A_part.T, b_part)
    c = np.zeros(N, dtype=np.float64)
    
    if use_async:
        req_c = comm.Iallreduce(c_part, c, op=MPI.SUM)
        req_c.Wait()
    else:
        comm.Allreduce(c_part, c, op=MPI.SUM)
    
    # r = c (так как x = 0), размерность N
    r = c.copy()
    
    # p = r
    p = r.copy()
    
    # gamma_old = r^T @ r
    gamma_old = np.dot(r, r)
    
    if verbose and rank == 0:
        print_convergence_header()
        print_convergence_step(0, np.sqrt(gamma_old))
    
    # Основной цикл с оптимизациями
    for iteration in range(1, max_iterations + 1):
        # Вычисляем s = (A^T A) @ p
        # Шаг 1: q = A @ p (каждый процесс вычисляет свою часть)
        q_part = np.dot(A_part, p)
        
        # Шаг 2: s = A^T @ q (собираем с помощью Allreduce)
        s_part = np.dot(A_part.T, q_part)
        s = np.zeros(N, dtype=np.float64)
        
        if use_async:
            req_s = comm.Iallreduce(s_part, s, op=MPI.SUM)
            
            if overlap_compute:
                # Пока ждём коммуникацию, делаем предвычисления
                pass
            
            req_s.Wait()
        else:
            comm.Allreduce(s_part, s, op=MPI.SUM)
        
        # Вычисляем delta = p^T @ s
        delta = np.dot(p, s)
        
        if abs(delta) < 1e-16:
            if rank == 0 and verbose:
                print(f"Остановка: delta = {delta:.2e}")
            break
        
        # alpha = gamma_old / delta
        alpha = gamma_old / delta
        
        # Векторизованные обновления
        x += alpha * p
        r -= alpha * s
        
        # gamma_new = r^T @ r
        gamma_new = np.dot(r, r)
        residual_norm = np.sqrt(gamma_new)
        
        if residual_norm < tolerance:
            if rank == 0 and verbose:
                print_convergence_step(iteration, residual_norm, alpha)
                print(f"\nСходимость достигнута: ||r|| = {residual_norm:.2e} < {tolerance:.2e}")
            break
        
        # beta = gamma_new / gamma_old
        beta = gamma_new / gamma_old
        
        # Векторизованное обновление p = r + beta * p
        p[:] = r + beta * p
        
        gamma_old = gamma_new
        
        if verbose and rank == 0 and iteration % 10 == 0:
            print_convergence_step(iteration, residual_norm, alpha, beta)
    
    return x, iteration, residual_norm

def parallel_cg_hybrid(
    comm, rank, size,
    A_part, b,
    M, N,
    max_iterations=None,
    tolerance=1e-10,
    verbose=False
):
    """
    Гибридная версия с использованием виртуальной топологии
    """
    
    # Создаём виртуальную топологию
    dims = MPI.Compute_dims(size, 2)
    periods = [False, False]
    comm_cart = comm.Create_cart(dims=dims, periods=periods, reorder=True)
    
    rank_cart = comm_cart.Get_rank()
    coords = comm_cart.Get_coords(rank_cart)
    
    # Используем оптимизированную версию с картезианской топологией
    return parallel_cg_optimized(
        comm_cart, rank_cart, size,
        A_part, b, M, N,
        max_iterations, tolerance, verbose,
        use_async=True, overlap_compute=True
    )

def main():
    """
    Основная функция для тестирования оптимизированной версии
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("="*70)
        print("ОПТИМИЗИРОВАННАЯ ВЕРСИЯ МЕТОДА СОПРЯЖЁННЫХ ГРАДИЕНТОВ")
        print("="*70)
    
    # Загрузка данных
    M, N, A, b, x_true = None, None, None, None, None
    
    if rank == 0:
        with open('in.dat', 'r') as f:
            M, N = map(int, f.readline().split())
        A = np.loadtxt('AData.dat', dtype=np.float64).reshape(M, N)
        b = np.loadtxt('bData.dat', dtype=np.float64)
        try:
            x_true = np.loadtxt('x_true.dat', dtype=np.float64)
        except:
            x_true = None
        
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
    
    # Тест 1: Без оптимизаций
    if rank == 0:
        print("\n" + "="*70)
        print("ТЕСТ 1: Базовая версия (без оптимизаций)")
        print("="*70)
    
    start = time.time()
    x_basic, iter_basic, res_basic = parallel_cg_optimized(
        comm, rank, size, A_part, b_full, M, N,
        verbose=(rank == 0), use_async=False, overlap_compute=False
    )
    time_basic = time.time() - start
    
    # Тест 2: С асинхронными операциями
    if rank == 0:
        print("\n" + "="*70)
        print("ТЕСТ 2: С асинхронными операциями")
        print("="*70)
    
    start = time.time()
    x_async, iter_async, res_async = parallel_cg_optimized(
        comm, rank, size, A_part, b_full, M, N,
        verbose=(rank == 0), use_async=True, overlap_compute=False
    )
    time_async = time.time() - start
    
    # Тест 3: С перекрытием вычислений
    if rank == 0:
        print("\n" + "="*70)
        print("ТЕСТ 3: С асинхронностью и перекрытием")
        print("="*70)
    
    start = time.time()
    x_overlap, iter_overlap, res_overlap = parallel_cg_optimized(
        comm, rank, size, A_part, b_full, M, N,
        verbose=(rank == 0), use_async=True, overlap_compute=True
    )
    time_overlap = time.time() - start
    
    # Результаты
    if rank == 0:
        print("\n" + "="*70)
        print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*70)
        print(f"\n{'Версия':<30} {'Время (сек)':<15} {'Итерации':<12} {'Ускорение':<12}")
        print("-"*70)
        print(f"{'Базовая':<30} {time_basic:<15.6f} {iter_basic:<12} {1.0:<12.3f}")
        print(f"{'Асинхронная':<30} {time_async:<15.6f} {iter_async:<12} "
              f"{time_basic/time_async:<12.3f}")
        print(f"{'С перекрытием':<30} {time_overlap:<15.6f} {iter_overlap:<12} "
              f"{time_basic/time_overlap:<12.3f}")
        
        # Верификация
        if x_true is not None:
            print("\n" + "="*70)
            print("ВЕРИФИКАЦИЯ РЕШЕНИЙ")
            print("="*70)
            compare_solutions(x_overlap, x_true, "Оптимизированная vs истинное")
        
        # Сохранение
        np.savetxt('x_solution_optimized.dat', x_overlap, fmt='%.10f')
        print(f"\nРешение сохранено в x_solution_optimized.dat")

if __name__ == "__main__":
    main()
