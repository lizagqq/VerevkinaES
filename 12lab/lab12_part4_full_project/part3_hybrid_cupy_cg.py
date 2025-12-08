#!/usr/bin/env python3
"""
Часть 4.3: Гибридный метод сопряженных градиентов (MPI + CuPy симуляция)
Для оценки "ОТЛИЧНО"
"""
from mpi4py import MPI
import numpy as np
# import cupy as cp  # В реальности
import time

class HybridCGSolver:
    def __init__(self, comm, device_A, host_b, global_cols):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Данные (в реальности на GPU через CuPy)
        self.device_A = device_A  # cp.asarray(host_A)
        self.device_b = host_b    # cp.asarray(host_b)
        self.global_cols = global_cols
        
        # Вспомогательные векторы
        self.device_x = np.zeros(global_cols, dtype=np.float32)
        self.device_r = np.zeros_like(host_b)
        self.device_p = np.zeros(global_cols, dtype=np.float32)
        self.device_Ap = np.zeros_like(host_b)
    
    def dot_product(self, vec1, vec2):
        """
        Скалярное произведение с редукцией
        В реальности: local_dot = cp.dot(vec1, vec2)
        """
        local_dot = np.dot(vec1, vec2).astype(np.float32)
        global_dot = np.array(0.0, dtype=np.float32)
        
        # Глобальная редукция через MPI
        self.comm.Allreduce([local_dot, MPI.FLOAT], 
                           [global_dot, MPI.FLOAT], op=MPI.SUM)
        
        return global_dot
    
    def mat_vec_product(self, vec):
        """
        Умножение матрицы на вектор
        В реальности: return cp.dot(self.device_A, vec)
        """
        return np.dot(self.device_A, vec)
    
    def solve(self, max_iter=100, tolerance=1e-6):
        """Метод сопряженных градиентов"""
        # Инициализация: r = b - A*x
        self.device_r = self.device_b - self.mat_vec_product(self.device_x)
        self.device_p = self.device_r.copy()
        
        rsold = self.dot_product(self.device_r, self.device_r)
        
        for iteration in range(max_iter):
            # Ap = A * p
            self.device_Ap = self.mat_vec_product(self.device_p)
            
            # alpha = rsold / (p^T * Ap)
            pAp = self.dot_product(self.device_p, self.device_Ap)
            
            if abs(pAp) < 1e-15:
                if self.rank == 0:
                    print(f"Warning: pAp too small at iteration {iteration}")
                break
            
            alpha = rsold / pAp
            
            # x = x + alpha * p
            self.device_x += alpha * self.device_p
            
            # r = r - alpha * Ap
            self.device_r -= alpha * self.device_Ap
            
            # rsnew = r^T * r
            rsnew = self.dot_product(self.device_r, self.device_r)
            
            # Проверка сходимости
            if rsnew < tolerance:
                if self.rank == 0:
                    print(f"Converged in {iteration+1} iterations, residual: {np.sqrt(rsnew):.6e}")
                break
            
            # beta = rsnew / rsold
            beta = rsnew / rsold
            
            # p = r + beta * p
            self.device_p = self.device_r + beta * self.device_p
            
            rsold = rsnew
        
        # В реальности: return cp.asnumpy(self.device_x)
        return self.device_x

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Параметры задачи
    local_rows = 1000
    global_cols = 1000
    
    if rank == 0:
        print("="*70)
        print("HYBRID CONJUGATE GRADIENT METHOD (MPI + CUPY)")
        print("="*70)
        print(f"Problem size: {local_rows*size} x {global_cols}")
        print(f"MPI processes: {size}")
        print(f"Local rows per process: {local_rows}")
        print("="*70)
    
    # Генерация тестовых данных (диагональное преобладание)
    np.random.seed(rank * 100)
    host_A = np.random.rand(local_rows, global_cols).astype(np.float32) * 0.1
    
    # Усиление диагонали для устойчивости
    for i in range(min(local_rows, global_cols)):
        if rank * local_rows + i < global_cols:
            host_A[i, rank * local_rows + i] += 5.0
    
    host_b = np.random.rand(local_rows).astype(np.float32)
    
    # Перенос на GPU (в реальности: device_A = cp.asarray(host_A))
    device_A = host_A
    
    # Создание решателя
    solver = HybridCGSolver(comm, device_A, host_b, global_cols)
    
    # Решение системы
    if rank == 0:
        print("\nSolving system...")
    
    comm.Barrier()
    start_time = MPI.Wtime()
    
    x_solution = solver.solve(max_iter=100, tolerance=1e-6)
    
    comm.Barrier()
    end_time = MPI.Wtime()
    
    elapsed = end_time - start_time
    
    if rank == 0:
        print(f"\n" + "="*70)
        print(f"Hybrid CG solution time: {elapsed:.6f} seconds")
        
        # Проверка решения
        residual = np.linalg.norm(host_b - np.dot(host_A, x_solution))
        print(f"Final residual: {residual:.4e}")
        print(f"Solution (first 5 elements): {x_solution[:5]}")
        print("="*70)
    
    return elapsed

if __name__ == "__main__":
    elapsed = main()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\nNote: Simulated CuPy operations")
        print("Real CuPy on GPU would be ~10x faster")
