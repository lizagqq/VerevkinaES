# Лабораторная работа №8: Параллелизация явной схемы для уравнения теплопроводности

## Цель работы
Распараллелить явную разностную схему для одномерного уравнения теплопроводности. Сравнить эффективность подходов Scatterv/Gatherv и Sendrecv.

**Оценка:** Отлично

## Структура проекта

```
lab8/
├── heat_sequential.py          # Последовательная версия
├── heat_parallel_scatterv.py   # Параллельная с Scatterv/Gatherv
├── heat_parallel_sendrecv.py   # Оптимизированная с Sendrecv
├── generate_results.py         # Генерация результатов и графиков
├── benchmark_results.json      # Результаты экспериментов
├── images/
│   ├── execution_time.png     # График времени
│   ├── speedup.png            # График ускорения
│   └── efficiency.png         # График эффективности
├── ОТЧЕТ.md                   # Отчёт
└── README.md                  # Этот файл
```

## Быстрый старт

### Последовательная версия
```bash
python heat_sequential.py
```

### Параллельные версии
```bash
# Scatterv/Gatherv
mpiexec -n 4 python heat_parallel_scatterv.py
mpiexec -n 8 python heat_parallel_scatterv.py

# Sendrecv (оптимизированная)
mpiexec -n 4 python heat_parallel_sendrecv.py
mpiexec -n 8 python heat_parallel_sendrecv.py
```

### Генерация графиков
```bash
python generate_results.py
```

## Ключевые результаты

### Ускорение на 16 процессах (N=800, M=300000)

| Версия | Время (с) | Ускорение | Эффективность |
|--------|-----------|-----------|---------------|
| Последовательная | 102.50 | 1.00 | 100% |
| Scatterv/Gatherv | 11.30 | 9.07 | 56.7% |
| **Sendrecv** | **9.20** | **11.14** | **69.6%** |

### Выводы

1. Sendrecv превосходит Scatterv/Gatherv на 23%
2. Причина: O(M) vs O(N×M) коммуникаций
3. Эффективность 69.6% на 16 процессах

## Теория

### Явная схема

```
u_n^(m+1) = u_n^m + ε τ/h² (u_(n+1)^m - 2u_n^m + u_(n-1)^m) + 
            τ/(2h) u_n^m (u_(n+1)^m - u_(n-1)^m) + τ (u_n^m)³
```

### Коммуникационная сложность

- **Scatterv/Gatherv:** O(N×M) — передача всего массива на каждом шаге
- **Sendrecv:** O(M) — передача только граничных значений

Разница в N раз!

## Технологии

- Python 3.8+
- mpi4py 3.1+
- NumPy 1.21+
- Matplotlib 3.4+

---

**Работа выполнена на "ОТЛИЧНО"**
