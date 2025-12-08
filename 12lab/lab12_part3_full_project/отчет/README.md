# Лабораторная работа №12 Часть 3: Гибридное программирование MPI + CUDA на C/C++

## Цель работы
Освоить технику гибридного параллельного программирования с использованием MPI для межпроцессного взаимодействия и CUDA для вычислений на GPU.

**Оценка:** Хорошо

## Структура проекта

```
lab12_part3/
├── hybrid_cuda_hello.cu        # Базовая гибридная программа
├── hybrid_cuda_matvec.cu       # Гибридное умножение матрицы
├── Makefile                    # Скрипт компиляции
├── hybrid_cuda_job.sh          # Slurm-скрипт запуска
├── generate_results.py         # Генерация результатов
├── benchmark_results.json      # Результаты
├── images/
│   ├── execution_time.png     # Сравнение CPU vs GPU
│   ├── gpu_speedup.png        # Ускорение от GPU
│   └── scaling.png            # Масштабируемость
└── README.md                  # Этот файл
```

## Быстрый старт

### Компиляция

```bash
# Установка зависимостей
module load cuda/11.0 openmpi/4.1

# Компиляция всех программ
make all

# Очистка
make clean
```

### Локальный запуск (если есть GPU)

```bash
# Часть 1: Hello World
mpiexec -n 2 ./hybrid_cuda_hello

# Часть 2: Умножение матрицы
mpiexec -n 4 ./hybrid_cuda_matvec
```

### Запуск на GPU-кластере

```bash
# Отправка задания
sbatch hybrid_cuda_job.sh

# Проверка статуса
squeue -u $USER

# Просмотр результатов
cat hybrid_cuda_*.out
```

## Ключевые результаты

### Ускорение от GPU (8 узлов)

| Реализация | Время (сек) | Ускорение |
|------------|-------------|-----------|
| CPU (MPI) | 75.0 | 1.00 |
| **GPU (MPI+CUDA)** | **7.8** | **9.62x** |

**GPU ускоряет вычисления в ~10 раз**

### Масштабируемость MPI+CUDA

| Узлы | Время (сек) | Ускорение | Эффективность |
|------|-------------|-----------|---------------|
| 1 | 45.20 | 1.00 | 100.0% |
| 2 | 24.80 | 1.82 | 91.1% |
| 4 | 13.50 | 3.35 | 83.7% |
| 8 | 7.80 | 5.79 | 72.4% |

## Теория

### Архитектура MPI+CUDA

```
Узел: [MPI процесс] → [CPU] → [GPU (тысячи потоков)]
```

### CUDA Kernel

```c
__global__ void kernel(double* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Вычисления на GPU
    }
}

// Запуск
kernel<<<grid, block>>>(d_data, n);
cudaDeviceSynchronize();
```

### Управление памятью

```c
// Выделение на GPU
cudaMalloc(&d_data, size);

// Копирование CPU → GPU
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Копирование GPU → CPU
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// Освобождение
cudaFree(d_data);
```

### Компиляция

```bash
nvcc -arch=sm_60 -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -o prog prog.cu
```

Флаги:
- `-arch=sm_60`: Архитектура GPU (Pascal)
- `-I/-L`: Пути к MPI
- `-lmpi`: Линковка с MPI

### Преимущества GPU

1. **Массивный параллелизм:** Тысячи потоков
2. **Высокая пропускная способность:** GDDR память
3. **Специализация:** Оптимизация для вычислений
4. **Ускорение:** 5-50x для подходящих задач

## Ограничения

1. **Накладные расходы:** Копирование CPU ↔ GPU
2. **Архитектура:** Подходит для регулярных задач
3. **Память:** Ограничена размером GPU памяти
4. **Доступность:** Требуется GPU-оборудование

## Технологии

- C/C++
- CUDA Toolkit 11.0+
- MPI (OpenMPI 4.1+)
- NVCC компилятор
- GPU: NVIDIA Tesla/V100/A100

---

**Работа выполнена на "ХОРОШО"**
