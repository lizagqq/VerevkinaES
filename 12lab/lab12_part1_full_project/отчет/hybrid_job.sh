#!/bin/bash
#SBATCH --job-name=hybrid_cg
#SBATCH --partition=test
#SBATCH --time=0:15:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --output=hybrid_cg_%j.out
#SBATCH --error=hybrid_cg_%j.err

# Настройка окружения
module load python/3.8
module load openmpi/4.1

# Активация виртуального окружения (если используется)
# source /path/to/venv/bin/activate

# Установка числа потоков OpenMP
export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=14

# Информация о конфигурации
echo "========================================"
echo "Hybrid MPI+OpenMP Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "========================================"

# Запуск Часть 1: Умножение матрицы на вектор
echo ""
echo "Running Part 1: Hybrid matrix-vector multiplication..."
srun --mpi=pmi2 python part1_hybrid_matvec.py

# Запуск Часть 2: Метод сопряженных градиентов
echo ""
echo "Running Part 2: Hybrid conjugate gradient method..."
srun --mpi=pmi2 python part2_hybrid_cg.py

# Запуск Часть 3: Сравнение с чистым MPI
echo ""
echo "Running Part 3: Comparison MPI vs Hybrid..."
srun --mpi=pmi2 python part3_comparison.py

# Запуск Часть 4: Исследование числа потоков
echo ""
echo "Running Part 4: Thread scaling analysis..."
srun --mpi=pmi2 python part4_thread_analysis.py

echo ""
echo "========================================"
echo "Job completed successfully"
echo "========================================"
