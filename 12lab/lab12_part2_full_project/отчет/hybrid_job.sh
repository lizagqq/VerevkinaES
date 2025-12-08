#!/bin/bash
#SBATCH --job-name=hybrid_mpi_omp
#SBATCH --partition=test
#SBATCH --time=0:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --output=hybrid_job_%j.out
#SBATCH --error=hybrid_job_%j.err

# Настройка окружения
module load gcc/9.3
module load openmpi/4.1

# Установка числа потоков OpenMP
export OMP_NUM_THREADS=14
export OMP_PROC_BIND=true
export OMP_PLACES=cores

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
srun ./hybrid_hello
echo ""

# Часть 2: Matrix-Vector Multiplication
echo "========================================"
echo "Part 2: Matrix-Vector Multiplication"
echo "========================================"
srun ./hybrid_matvec
echo ""

# Часть 3: Conjugate Gradient Method
echo "========================================"
echo "Part 3: Conjugate Gradient Method"
echo "========================================"
srun ./hybrid_cg
echo ""

# Часть 4: Сравнение различных конфигураций потоков
echo "========================================"
echo "Part 4: Thread Scaling Analysis"
echo "========================================"

for threads in 1 2 4 8 14; do
    echo ""
    echo "--- Testing with $threads threads ---"
    export OMP_NUM_THREADS=$threads
    srun ./hybrid_cg
done

echo ""
echo "========================================"
echo "Job completed successfully"
echo "========================================"
