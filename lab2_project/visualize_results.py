import json
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('images', exist_ok=True)

def load_results():
    """Загружает результаты бенчмарков"""
    with open('benchmark_results.json', 'r') as f:
        return json.load(f)

def plot_part1_results(results):
    """Строит графики для Части 1"""
    part1_data = results['part1']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sizes = list(part1_data.keys())
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        data = part1_data[size]
        
        procs = sorted([int(k) for k in data.keys()])
        times_par = [data[str(p)]['parallel_time'] for p in procs]
        times_seq = [data[str(p)]['sequential_time'] for p in procs]
        speedups = [data[str(p)]['speedup'] for p in procs]
        
        # График времени
        ax2 = ax.twinx()
        line1 = ax.plot(procs, times_par, 'o-', linewidth=2, markersize=8, 
                        label='Параллельное', color='blue')
        line2 = ax.plot(procs, times_seq, 's--', linewidth=2, markersize=8, 
                        label='Последовательное', color='green', alpha=0.7)
        
        # График ускорения на другой оси
        line3 = ax2.plot(procs, speedups, '^-', linewidth=2, markersize=8, 
                         label='Ускорение', color='red')
        
        ax.set_xlabel('Количество процессов', fontsize=11)
        ax.set_ylabel('Время (сек)', fontsize=11, color='black')
        ax2.set_ylabel('Ускорение', fontsize=11, color='red')
        ax.set_title(f'Размер вектора: {size.split("=")[1]}', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(procs)
        
        # Объединённая легенда
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('images/part1_results.png', dpi=300, bbox_inches='tight')
    print("График Части 1 сохранён: images/part1_results.png")
    plt.close()

def plot_part2_results(results):
    """Строит графики для Части 2"""
    part2_data = results['part2']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sizes = list(part2_data.keys())
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        data = part2_data[size]
        
        procs = sorted([int(k) for k in data.keys()])
        times_par = [data[str(p)]['parallel_time'] for p in procs]
        times_seq = [data[str(p)]['sequential_time'] for p in procs]
        speedups = [data[str(p)]['speedup'] for p in procs]
        
        # График времени
        ax2 = ax.twinx()
        line1 = ax.plot(procs, times_par, 'o-', linewidth=2, markersize=8, 
                        label='Параллельное', color='blue')
        line2 = ax.plot(procs, times_seq, 's--', linewidth=2, markersize=8, 
                        label='Последовательное', color='green', alpha=0.7)
        
        # График ускорения на другой оси
        line3 = ax2.plot(procs, speedups, '^-', linewidth=2, markersize=8, 
                         label='Ускорение', color='red')
        
        ax.set_xlabel('Количество процессов', fontsize=11)
        ax.set_ylabel('Время (сек)', fontsize=11, color='black')
        ax2.set_ylabel('Ускорение', fontsize=11, color='red')
        ax.set_title(f'Матрица: {size}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(procs)
        
        # Объединённая легенда
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('images/part2_results.png', dpi=300, bbox_inches='tight')
    print("График Части 2 сохранён: images/part2_results.png")
    plt.close()

def plot_speedup_comparison(results):
    """Сравнение ускорения для обеих частей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Часть 1
    part1_data = results['part1']
    for size, data in part1_data.items():
        procs = sorted([int(k) for k in data.keys()])
        speedups = [data[str(p)]['speedup'] for p in procs]
        ax1.plot(procs, speedups, 'o-', linewidth=2, markersize=8, 
                 label=f'{size.split("=")[1]} элементов')
    
    # Идеальное ускорение
    max_procs = max([int(k) for k in part1_data[list(part1_data.keys())[0]].keys()])
    ideal_procs = list(range(1, max_procs + 1))
    ax1.plot(ideal_procs, ideal_procs, '--', linewidth=2, alpha=0.7, 
             label='Идеальное', color='gray')
    
    ax1.set_xlabel('Количество процессов', fontsize=12)
    ax1.set_ylabel('Ускорение', fontsize=12)
    ax1.set_title('Часть 1: Скалярное произведение', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(ideal_procs)
    
    # Часть 2
    part2_data = results['part2']
    for size, data in part2_data.items():
        procs = sorted([int(k) for k in data.keys()])
        speedups = [data[str(p)]['speedup'] for p in procs]
        ax2.plot(procs, speedups, 'o-', linewidth=2, markersize=8, label=size)
    
    ax2.plot(ideal_procs, ideal_procs, '--', linewidth=2, alpha=0.7, 
             label='Идеальное', color='gray')
    
    ax2.set_xlabel('Количество процессов', fontsize=12)
    ax2.set_ylabel('Ускорение', fontsize=12)
    ax2.set_title('Часть 2: Умножение A.T @ x', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(ideal_procs)
    
    plt.tight_layout()
    plt.savefig('images/speedup_comparison.png', dpi=300, bbox_inches='tight')
    print("График сравнения ускорений сохранён: images/speedup_comparison.png")
    plt.close()

def plot_efficiency(results):
    """График эффективности"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Часть 1
    part1_data = results['part1']
    for size, data in part1_data.items():
        procs = sorted([int(k) for k in data.keys()])
        efficiencies = [data[str(p)]['efficiency'] * 100 for p in procs]
        ax1.plot(procs, efficiencies, 'o-', linewidth=2, markersize=8, 
                 label=f'{size.split("=")[1]} элементов')
    
    ax1.axhline(y=100, linestyle='--', linewidth=2, alpha=0.7, 
                color='gray', label='Идеальная (100%)')
    
    ax1.set_xlabel('Количество процессов', fontsize=12)
    ax1.set_ylabel('Эффективность (%)', fontsize=12)
    ax1.set_title('Часть 1: Скалярное произведение', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 110)
    
    # Часть 2
    part2_data = results['part2']
    for size, data in part2_data.items():
        procs = sorted([int(k) for k in data.keys()])
        efficiencies = [data[str(p)]['efficiency'] * 100 for p in procs]
        ax2.plot(procs, efficiencies, 'o-', linewidth=2, markersize=8, label=size)
    
    ax2.axhline(y=100, linestyle='--', linewidth=2, alpha=0.7, 
                color='gray', label='Идеальная (100%)')
    
    ax2.set_xlabel('Количество процессов', fontsize=12)
    ax2.set_ylabel('Эффективность (%)', fontsize=12)
    ax2.set_title('Часть 2: Умножение A.T @ x', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('images/efficiency.png', dpi=300, bbox_inches='tight')
    print("График эффективности сохранён: images/efficiency.png")
    plt.close()

def main():
    print("Загрузка результатов...")
    results = load_results()
    
    print("Создание графиков...")
    plot_part1_results(results)
    plot_part2_results(results)
    plot_speedup_comparison(results)
    plot_efficiency(results)
    
    print("\nВизуализация завершена!")

if __name__ == "__main__":
    main()
