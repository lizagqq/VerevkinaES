#!/bin/bash
#SBATCH --job-name=hybrid_mpi_cuda
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=hybrid_cuda_%j.out
#SBATCH --error=hybrid_cuda_%j.err

# Настройка окружения
module load cuda/11.0
module load openmpi/4.1

# Информация о конфигурации
echo "========================================"
echo "Hybrid MPI+CUDA Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: 1"
echo "========================================"
echo ""

# Компиляция программ
echo "Compiling programs..."
make clean
make all
echo ""

# Часть 1: Hello World
echo "========================================"
echo "Part 1: Hello World"
echo "========================================"
srun ./hybrid_cuda_hello
echo ""

# Часть 2: Matrix-Vector Multiplication
echo "========================================"
echo "Part 2: Matrix-Vector Multiplication"
echo "========================================"
srun ./hybrid_cuda_matvec
echo ""

# Тестирование на различном числе узлов
echo "========================================"
echo "Scaling Tests"
echo "========================================"

for nodes in 1 2 4; do
    echo ""
    echo "--- Testing with $nodes nodes ---"
    srun -N $nodes ./hybrid_cuda_matvec
done

echo ""
echo "========================================"
echo "Job completed successfully"
echo "========================================"
