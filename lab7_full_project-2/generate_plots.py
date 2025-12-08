#!/usr/bin/env python3
"""
Генерация графиков для отчёта
"""
import numpy as np
import matplotlib.pyplot as plt
import json

# Загрузка результатов
with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

# Данные
sizes = [100, 1000, 10000, 50000]
processes = [1, 2, 4, 8]

# Подготовка данных для графиков
times_seq = [results['sequential'][str(N)]['time'] for N in sizes]
times_parallel = {}
speedups = {}
efficiencies = {}

for P in [2, 4, 8]:
    times_parallel[P] = [results['parallel'][str(N)][str(P)]['time'] for N in sizes]
    speedups[P] = [results['parallel'][str(N)][str(P)]['speedup'] for N in sizes]
    efficiencies[P] = [results['parallel'][str(N)][str(P)]['efficiency'] * 100 for N in sizes]

# График 1: Время выполнения
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_seq, 'o-', label='Последовательно', linewidth=2, markersize=8)
for P in [2, 4, 8]:
    plt.plot(sizes, times_parallel[P], 'o-', label=f'{P} процесса', linewidth=2, markersize=8)
plt.xlabel('Размер системы N', fontsize=12)
plt.ylabel('Время выполнения (с)', fontsize=12)
plt.title('Зависимость времени выполнения от размера системы', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('images/execution_time.png', dpi=300, bbox_inches='tight')
plt.close()

# График 2: Ускорение
plt.figure(figsize=(10, 6))
for P in [2, 4, 8]:
    plt.plot(sizes, speedups[P], 'o-', label=f'{P} процесса', linewidth=2, markersize=8)
plt.plot(sizes, [2]*len(sizes), 'k--', alpha=0.3, label='Идеальное (P=2)')
plt.plot(sizes, [4]*len(sizes), 'k--', alpha=0.3, label='Идеальное (P=4)')
plt.plot(sizes, [8]*len(sizes), 'k--', alpha=0.3, label='Идеальное (P=8)')
plt.xlabel('Размер системы N', fontsize=12)
plt.ylabel('Ускорение (раз)', fontsize=12)
plt.title('Ускорение параллельного алгоритма', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()
plt.savefig('images/speedup.png', dpi=300, bbox_inches='tight')
plt.close()

# График 3: Эффективность
plt.figure(figsize=(10, 6))
for P in [2, 4, 8]:
    plt.plot(sizes, efficiencies[P], 'o-', label=f'{P} процесса', linewidth=2, markersize=8)
plt.axhline(y=100, color='k', linestyle='--', alpha=0.3, label='Идеальная (100%)')
plt.xlabel('Размер системы N', fontsize=12)
plt.ylabel('Эффективность (%)', fontsize=12)
plt.title('Эффективность распараллеливания', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig('images/efficiency.png', dpi=300, bbox_inches='tight')
plt.close()

print("Графики сохранены в директорию images/")
print("  - execution_time.png")
print("  - speedup.png")
print("  - efficiency.png")
