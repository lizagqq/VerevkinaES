"""
Визуализация результатов профилирования и бенчмаркинга
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Настройка стиля
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_results():
    """Загрузка результатов"""
    with open('../results/benchmark_summary.json', 'r') as f:
        return json.load(f)

def plot_execution_time(results):
    """График времени выполнения"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    test_configs = [
        ("test_500x100", "500×100"),
        ("test_1000x200", "1000×200"),
        ("test_2000x400", "2000×400")
    ]
    
    proc_counts = [1, 2, 4, 8]
    
    for idx, (prefix, label) in enumerate(test_configs):
        ax = axes[idx]
        
        baseline_times = [results['baseline'][prefix][str(np)]['total_time'] 
                         for np in proc_counts]
        optimized_times = [results['optimized'][prefix][str(np)]['total_time'] 
                          for np in proc_counts]
        
        x = np.arange(len(proc_counts))
        width = 0.35
        
        ax.bar(x - width/2, baseline_times, width, label='Базовая версия', 
               color='#E74C3C', alpha=0.8)
        ax.bar(x + width/2, optimized_times, width, label='Оптимизированная', 
               color='#27AE60', alpha=0.8)
        
        ax.set_xlabel('Количество процессов')
        ax.set_ylabel('Время выполнения (сек)')
        ax.set_title(f'Размер задачи: {label}')
        ax.set_xticks(x)
        ax.set_xticklabels(proc_counts)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../images/execution_time.png', dpi=300, bbox_inches='tight')
    print("Сохранён график: execution_time.png")
    plt.close()

def plot_speedup(results):
    """График ускорения"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    test_configs = [
        ("test_500x100", "500×100"),
        ("test_1000x200", "1000×200"),
        ("test_2000x400", "2000×400")
    ]
    
    proc_counts = [1, 2, 4, 8]
    
    ax.plot(proc_counts, proc_counts, 'k--', label='Идеальное ускорение', 
            linewidth=2, alpha=0.5)
    
    colors_baseline = ['#E74C3C', '#E67E22', '#F39C12']
    colors_optimized = ['#27AE60', '#2ECC71', '#16A085']
    
    for idx, (prefix, label) in enumerate(test_configs):
        baseline_times = [results['baseline'][prefix][str(np)]['total_time'] 
                         for np in proc_counts]
        baseline_speedup = [baseline_times[0] / t for t in baseline_times]
        
        ax.plot(proc_counts, baseline_speedup, 'o-', 
                color=colors_baseline[idx], 
                label=f'Базовая {label}',
                linewidth=2, markersize=8)
        
        optimized_times = [results['optimized'][prefix][str(np)]['total_time'] 
                          for np in proc_counts]
        optimized_speedup = [optimized_times[0] / t for t in optimized_times]
        
        ax.plot(proc_counts, optimized_speedup, 's-', 
                color=colors_optimized[idx], 
                label=f'Оптимизированная {label}',
                linewidth=2, markersize=8)
    
    ax.set_xlabel('Количество процессов', fontsize=12)
    ax.set_ylabel('Ускорение (Speedup)', fontsize=12)
    ax.set_title('Сравнение ускорения базовой и оптимизированной версий', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xticks(proc_counts)
    
    plt.tight_layout()
    plt.savefig('../images/speedup.png', dpi=300, bbox_inches='tight')
    print("Сохранён график: speedup.png")
    plt.close()

def plot_efficiency(results):
    """График эффективности"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    test_configs = [
        ("test_500x100", "500×100"),
        ("test_1000x200", "1000×200"),
        ("test_2000x400", "2000×400")
    ]
    
    proc_counts = [1, 2, 4, 8]
    
    ax.axhline(y=1.0, color='k', linestyle='--', label='Идеальная эффективность', 
               linewidth=2, alpha=0.5)
    
    colors_baseline = ['#E74C3C', '#E67E22', '#F39C12']
    colors_optimized = ['#27AE60', '#2ECC71', '#16A085']
    
    for idx, (prefix, label) in enumerate(test_configs):
        baseline_times = [results['baseline'][prefix][str(np)]['total_time'] 
                         for np in proc_counts]
        baseline_speedup = [baseline_times[0] / t for t in baseline_times]
        baseline_efficiency = [s / p for s, p in zip(baseline_speedup, proc_counts)]
        
        ax.plot(proc_counts, baseline_efficiency, 'o-', 
                color=colors_baseline[idx], 
                label=f'Базовая {label}',
                linewidth=2, markersize=8)
        
        optimized_times = [results['optimized'][prefix][str(np)]['total_time'] 
                          for np in proc_counts]
        optimized_speedup = [optimized_times[0] / t for t in optimized_times]
        optimized_efficiency = [s / p for s, p in zip(optimized_speedup, proc_counts)]
        
        ax.plot(proc_counts, optimized_efficiency, 's-', 
                color=colors_optimized[idx], 
                label=f'Оптимизированная {label}',
                linewidth=2, markersize=8)
    
    ax.set_xlabel('Количество процессов', fontsize=12)
    ax.set_ylabel('Эффективность (Efficiency)', fontsize=12)
    ax.set_title('Сравнение эффективности базовой и оптимизированной версий', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xticks(proc_counts)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('../images/efficiency.png', dpi=300, bbox_inches='tight')
    print("Сохранён график: efficiency.png")
    plt.close()

def plot_time_breakdown(results):
    """График распределения времени"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    proc_counts = [1, 2, 4, 8]
    prefix = "test_1000x200"
    
    for idx, num_procs in enumerate(proc_counts):
        ax = axes[idx]
        
        baseline_times = results['baseline'][prefix][str(num_procs)]['times']
        baseline_labels = list(baseline_times.keys())
        baseline_values = list(baseline_times.values())
        
        optimized_times = results['optimized'][prefix][str(num_procs)]['times']
        optimized_labels = list(optimized_times.keys())
        optimized_values = list(optimized_times.values())
        
        x = np.arange(max(len(baseline_labels), len(optimized_labels)))
        width = 0.35
        
        ax.bar(x - width/2, baseline_values[:len(baseline_labels)], width, 
               label='Базовая', color='#E74C3C', alpha=0.8)
        ax.bar(x + width/2, optimized_values[:len(optimized_labels)], width, 
               label='Оптимизированная', color='#27AE60', alpha=0.8)
        
        ax.set_xlabel('Тип операции')
        ax.set_ylabel('Время (сек)')
        ax.set_title(f'{num_procs} процесс(ов)')
        all_labels = list(set(baseline_labels + optimized_labels))
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Распределение времени по операциям (задача 1000×200)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../images/time_breakdown.png', dpi=300, bbox_inches='tight')
    print("Сохранён график: time_breakdown.png")
    plt.close()

def plot_improvement_heatmap(results):
    """Тепловая карта улучшений"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    test_configs = [
        ("test_500x100", "500×100"),
        ("test_1000x200", "1000×200"),
        ("test_2000x400", "2000×400")
    ]
    
    proc_counts = [1, 2, 4, 8]
    
    improvement_matrix = []
    
    for prefix, label in test_configs:
        row = []
        for num_procs in proc_counts:
            baseline_time = results['baseline'][prefix][str(num_procs)]['total_time']
            optimized_time = results['optimized'][prefix][str(num_procs)]['total_time']
            improvement = ((baseline_time - optimized_time) / baseline_time) * 100
            row.append(improvement)
        improvement_matrix.append(row)
    
    im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=30)
    
    ax.set_xticks(np.arange(len(proc_counts)))
    ax.set_yticks(np.arange(len(test_configs)))
    ax.set_xticklabels(proc_counts)
    ax.set_yticklabels([label for _, label in test_configs])
    
    for i in range(len(test_configs)):
        for j in range(len(proc_counts)):
            text = ax.text(j, i, f'{improvement_matrix[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xlabel('Количество процессов', fontsize=12)
    ax.set_ylabel('Размер задачи', fontsize=12)
    ax.set_title('Улучшение производительности (%)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Улучшение (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('../images/improvement_heatmap.png', dpi=300, bbox_inches='tight')
    print("Сохранён график: improvement_heatmap.png")
    plt.close()

def plot_scalability_comparison(results):
    """Сравнение масштабируемости"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    prefix = "test_2000x400"
    proc_counts = [1, 2, 4, 8]
    
    baseline_times = [results['baseline'][prefix][str(num_procs)]['total_time'] 
                     for num_procs in proc_counts]
    baseline_speedup = [baseline_times[0] / t for t in baseline_times]
    baseline_efficiency = [s / p * 100 for s, p in zip(baseline_speedup, proc_counts)]
    
    optimized_times = [results['optimized'][prefix][str(num_procs)]['total_time'] 
                      for num_procs in proc_counts]
    optimized_speedup = [optimized_times[0] / t for t in optimized_times]
    optimized_efficiency = [s / p * 100 for s, p in zip(optimized_speedup, proc_counts)]
    
    x = np.arange(len(proc_counts))
    width = 0.35
    
    ax.bar(x - width/2, baseline_efficiency, width, 
           label='Базовая версия', color='#E74C3C', alpha=0.8)
    ax.bar(x + width/2, optimized_efficiency, width, 
           label='Оптимизированная версия', color='#27AE60', alpha=0.8)
    
    for i, (b, o) in enumerate(zip(baseline_efficiency, optimized_efficiency)):
        ax.text(i - width/2, b + 1, f'{b:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, o + 1, f'{o:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Количество процессов', fontsize=12)
    ax.set_ylabel('Параллельная эффективность (%)', fontsize=12)
    ax.set_title('Сравнение параллельной эффективности (задача 2000×400)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(proc_counts)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 110])
    
    plt.tight_layout()
    plt.savefig('../images/scalability_comparison.png', dpi=300, bbox_inches='tight')
    print("Сохранён график: scalability_comparison.png")
    plt.close()

def main():
    print("Загрузка результатов...")
    results = load_results()
    
    print("\nСоздание графиков...")
    plot_execution_time(results)
    plot_speedup(results)
    plot_efficiency(results)
    plot_time_breakdown(results)
    plot_improvement_heatmap(results)
    plot_scalability_comparison(results)
    
    print("\nВсе графики созданы и сохранены в ../images/")

if __name__ == "__main__":
    main()
