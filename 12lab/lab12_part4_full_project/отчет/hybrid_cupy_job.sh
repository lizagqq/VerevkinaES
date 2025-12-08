#!/bin/bash
#SBATCH --job-name=hybrid_cupy
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=hybrid_cupy_%j.out
#SBATCH --error=hybrid_cupy_%j.err

# Настройка окружения
module load python/3.8
module load cuda/11.0
module load openmpi/4.1

# Активация виртуального окружения
source ~/myenv/bin/activate

# Установка CuPy (если не установлен)
# pip install cupy-cuda110

# Информация о конфигурации
echo "========================================"
echo "Hybrid MPI+CuPy Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: 1"
echo "========================================"
echo ""

# Часть 1: Hello World
echo "========================================"
echo "Part 1: Hello World"
echo "========================================"
srun python part1_hybrid_cupy_hello.py
echo ""

# Часть 2: Matrix-Vector Multiplication
echo "========================================"
echo "Part 2: Matrix-Vector Multiplication"
echo "========================================"
srun python part2_hybrid_cupy_matvec.py
echo ""

# Часть 3: Conjugate Gradient Method
echo "========================================"
echo "Part 3: Conjugate Gradient Method"
echo "========================================"
srun python part3_hybrid_cupy_cg.py
echo ""

# Часть 4: Comparative Analysis
echo "========================================"
echo "Part 4: NumPy vs CuPy Comparison"
echo "========================================"
srun python part4_comparison.py
echo ""

# Генерация результатов и графиков
echo "========================================"
echo "Generating Results and Plots"
echo "========================================"
python generate_results.py

echo ""
echo "========================================"
echo "Job completed successfully"
echo "========================================"
