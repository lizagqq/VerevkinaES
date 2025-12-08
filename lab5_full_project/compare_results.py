#!/usr/bin/env python3
"""
Сравнительный анализ одномерной и двумерной декомпозиции
"""

import json
import numpy as np

# Синтетические результаты для демонстрации
results = {
    "1D_row": {  # Одномерная по строкам (Lab1)
        "small_100x100": {
            "4": {"time": 0.0823, "speedup": 2.91, "efficiency": 0.728},
            "9": {"time": 0.0456, "speedup": 5.25, "efficiency": 0.583},
            "16": {"time": 0.0334, "speedup": 7.16, "efficiency": 0.448}
        },
        "medium_500x500": {
            "4": {"time": 1.845, "speedup": 3.12, "efficiency": 0.780},
            "9": {"time": 0.987, "speedup": 5.83, "efficiency": 0.648},
            "16": {"time": 0.623, "speedup": 9.24, "efficiency": 0.578}
        },
        "large_1000x1000": {
            "4": {"time": 7.234, "speedup": 3.24, "efficiency": 0.810},
            "9": {"time": 3.678, "speedup": 6.37, "efficiency": 0.708},
            "16": {"time": 2.145, "speedup": 10.92, "efficiency": 0.683}
        }
    },
    "2D_block": {  # Двумерная блочная (Lab5)
        "small_100x100": {
            "4": {"time": 0.0745, "speedup": 3.21, "efficiency": 0.803},
            "9": {"time": 0.0389, "speedup": 6.15, "efficiency": 0.683},
            "16": {"time": 0.0267, "speedup": 8.96, "efficiency": 0.560}
        },
        "medium_500x500": {
            "4": {"time": 1.623, "speedup": 3.54, "efficiency": 0.886},
            "9": {"time": 0.812, "speedup": 7.08, "efficiency": 0.787},
            "16": {"time": 0.478, "speedup": 12.03, "efficiency": 0.752}
        },
        "large_1000x1000": {
            "4": {"time": 6.345, "speedup": 3.69, "efficiency": 0.923},
            "9": {"time": 2.987, "speedup": 7.84, "efficiency": 0.871},
            "16": {"time": 1.634, "speedup": 14.33, "efficiency": 0.896}
        }
    }
}

def print_comparison_table():
    """Выводит таблицу сравнения"""
    print("\n" + "="*80)
    print("СРАВНЕНИЕ ОДНОМЕРНОЙ И ДВУМЕРНОЙ ДЕКОМПОЗИЦИИ")
    print("="*80)
    
    datasets = ["small_100x100", "medium_500x500", "large_1000x1000"]
    procs = ["4", "9", "16"]
    
    for dataset in datasets:
        print(f"\n{dataset.upper().replace('_', ' ')}:")
        print("-"*80)
        print(f"{'Процессы':<10} {'Метод':<15} {'Время (с)':<12} {'Ускорение':<12} {'Эффективность':<15}")
        print("-"*80)
        
        for p in procs:
            # 1D
            r1d = results["1D_row"][dataset][p]
            print(f"{p:<10} {'1D (строки)':<15} {r1d['time']:<12.4f} {r1d['speedup']:<12.2f} {r1d['efficiency']*100:<14.1f}%")
            
            # 2D
            r2d = results["2D_block"][dataset][p]
            improvement = ((r1d['time'] - r2d['time']) / r1d['time']) * 100
            print(f"{'':<10} {'2D (блоки)':<15} {r2d['time']:<12.4f} {r2d['speedup']:<12.2f} {r2d['efficiency']*100:<14.1f}%")
            print(f"{'':<10} {'Улучшение:':<15} {improvement:>11.1f}%")
            print()

def calculate_communication_volume():
    """Анализ объёма коммуникаций"""
    print("\n" + "="*80)
    print("АНАЛИЗ ОБЪЁМА КОММУНИКАЦИЙ")
    print("="*80)
    
    # M = N для квадратных матриц
    sizes = [100, 500, 1000]
    
    print(f"\n{'Размер M×N':<15} {'1D (Bcast)':<20} {'2D (Bcast+Reduce)':<25} {'Выигрыш':<15}")
    print("-"*80)
    
    for N in sizes:
        M = N
        # 1D: Bcast вектора x (N элементов)
        comm_1d = N * 8  # байт (double)
        
        # 2D: Bcast по строкам + редукция по столбцам
        # Предположим P = 16, сетка 4x4
        P = 16
        sqrt_P = 4
        comm_2d = (N // sqrt_P) * 8 * sqrt_P  # Bcast части вектора
        comm_2d += (M // sqrt_P) * 8 * sqrt_P  # Reduce частичных результатов
        
        ratio = comm_1d / comm_2d
        
        print(f"{M}×{N:<11} {comm_1d/1024:<18.1f} KB {comm_2d/1024:<23.1f} KB {ratio:<14.2f}x")

if __name__ == "__main__":
    # Сохранение результатов
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print_comparison_table()
    calculate_communication_volume()
    
    print("\n✓ Результаты сохранены в comparison_results.json")
