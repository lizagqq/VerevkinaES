import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Создаём директорию для изображений
os.makedirs('images', exist_ok=True)

def load_results():
    """Загружает результаты бенчмарков"""
    with open('benchmark_results.json', 'r') as f:
        return json.load(f)

def plot_execution_time(results):
    """Строит график времени выполнения"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sizes = list(results.keys())
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        data = results[size]
        
        # Последовательное время
        seq_time = data['sequential']
        
        # Параллельные времена
        proc_counts = sorted([int(k) for k in data['parallel'].keys()])
        par_times = [data['parallel'][str(p)]['time'] for p in proc_counts]
        
        # Добавляем последовательное время для сравнения
        all_procs = [1] + proc_counts
        all_times = [seq_time] + par_times
        
        ax.plot(all_procs, all_times, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Количество процессов', fontsize=12)
        ax.set_ylabel('Время выполнения (сек)', fontsize=12)
        ax.set_title(f'Размер матрицы: {size}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_procs)
    
    plt.tight_layout()
    plt.savefig('images/execution_time.png', dpi=300, bbox_inches='tight')
    print("График времени выполнения сохранён: images/execution_time.png")
    plt.close()

def plot_speedup(results):
    """Строит график ускорения"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sizes = list(results.keys())
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        data = results[size]
        
        # Параллельные ускорения
        proc_counts = sorted([int(k) for k in data['parallel'].keys()])
        speedups = [data['parallel'][str(p)]['speedup'] for p in proc_counts]
        
        # Добавляем точку для 1 процесса
        all_procs = [1] + proc_counts
        all_speedups = [1.0] + speedups
        
        # Идеальное ускорение
        ideal_speedup = np.array(all_procs)
        
        ax.plot(all_procs, all_speedups, 'o-', linewidth=2, markersize=8, label='Реальное')
        ax.plot(all_procs, ideal_speedup, '--', linewidth=2, alpha=0.7, label='Идеальное')
        
        ax.set_xlabel('Количество процессов', fontsize=12)
        ax.set_ylabel('Ускорение', fontsize=12)
        ax.set_title(f'Размер матрицы: {size}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(all_procs)
    
    plt.tight_layout()
    plt.savefig('images/speedup.png', dpi=300, bbox_inches='tight')
    print("График ускорения сохранён: images/speedup.png")
    plt.close()

def plot_efficiency(results):
    """Строит график эффективности"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sizes = list(results.keys())
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        data = results[size]
        
        # Параллельные эффективности
        proc_counts = sorted([int(k) for k in data['parallel'].keys()])
        efficiencies = [data['parallel'][str(p)]['efficiency'] * 100 for p in proc_counts]
        
        # Добавляем точку для 1 процесса
        all_procs = [1] + proc_counts
        all_effs = [100.0] + efficiencies
        
        # Идеальная эффективность (100%)
        ideal_eff = [100] * len(all_procs)
        
        ax.plot(all_procs, all_effs, 'o-', linewidth=2, markersize=8, label='Реальная')
        ax.plot(all_procs, ideal_eff, '--', linewidth=2, alpha=0.7, label='Идеальная')
        
        ax.set_xlabel('Количество процессов', fontsize=12)
        ax.set_ylabel('Эффективность (%)', fontsize=12)
        ax.set_title(f'Размер матрицы: {size}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(all_procs)
        ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('images/efficiency.png', dpi=300, bbox_inches='tight')
    print("График эффективности сохранён: images/efficiency.png")
    plt.close()

def create_summary_table(results):
    """Создаёт сводную таблицу результатов"""
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    for size, data in results.items():
        print(f"\nРазмер матрицы: {size}")
        print("-" * 80)
        print(f"{'Процессы':<12} {'Время (сек)':<15} {'Ускорение':<15} {'Эффективность':<15}")
        print("-" * 80)
        
        # Последовательная версия
        seq_time = data['sequential']
        print(f"{'1 (seq)':<12} {seq_time:<15.6f} {'1.00':<15} {'100.0%':<15}")
        
        # Параллельные версии
        for proc_count in sorted([int(k) for k in data['parallel'].keys()]):
            par_data = data['parallel'][str(proc_count)]
            print(f"{proc_count:<12} {par_data['time']:<15.6f} "
                  f"{par_data['speedup']:<15.2f} {par_data['efficiency']*100:<15.1f}%")

def main():
    """Главная функция для визуализации"""
    print("Загрузка результатов бенчмарков...")
    results = load_results()
    
    print("Создание графиков...")
    plot_execution_time(results)
    plot_speedup(results)
    plot_efficiency(results)
    
    create_summary_table(results)
    
    print("\n" + "=" * 80)
    print("Визуализация завершена!")
    print("=" * 80)

if __name__ == "__main__":
    main()
