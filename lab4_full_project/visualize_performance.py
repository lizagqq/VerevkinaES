#!/usr/bin/env python3
"""
Визуализация результатов анализа производительности
Строит графики ускорения, эффективности, сильной/слабой масштабируемости
"""

import json
import matplotlib
matplotlib.use('Agg')  # Backend для сохранения без дисплея
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class PerformanceVisualizer:
    """Класс для визуализации метрик производительности"""
    
    def __init__(self, results_file='synthetic_results.json', output_dir='plots'):
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        with open(results_file, 'r') as f:
            self.results = json.load(f)
    
    def plot_strong_scaling_time(self):
        """График времени выполнения для сильной масштабируемости"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Сильная масштабируемость: Время выполнения', fontsize=14, fontweight='bold')
        
        datasets = ['setA', 'setB', 'setC']
        titles = ['Набор A (N=200, M=20M)', 'Набор B (N=500, M=8M)', 'Набор C (N=1000, M=2M)']
        
        for idx, (dataset, title) in enumerate(zip(datasets, titles)):
            ax = axes[idx]
            
            for exp_name, data in self.results['experiments'].items():
                if dataset not in data:
                    continue
                
                proc_results = data[dataset]
                procs = sorted([int(p) for p in proc_results.keys()])
                times = [proc_results[str(p)]['time'] for p in procs]
                
                ax.plot(procs, times, 'o-', label=exp_name, linewidth=2, markersize=8)
            
            ax.set_xlabel('Количество процессов', fontsize=11)
            ax.set_ylabel('Время (сек)', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        output_file = self.output_dir / 'strong_scaling_time.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ График сохранён: {output_file}")
    
    def plot_strong_scaling_speedup(self):
        """График ускорения для сильной масштабируемости"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Сильная масштабируемость: Ускорение', fontsize=14, fontweight='bold')
        
        datasets = ['setA', 'setB', 'setC']
        titles = ['Набор A', 'Набор B', 'Набор C']
        
        for idx, (dataset, title) in enumerate(zip(datasets, titles)):
            ax = axes[idx]
            
            # Идеальное ускорение
            base_procs = None
            for exp_name, data in self.results['experiments'].items():
                if dataset in data:
                    base_procs = min([int(p) for p in data[dataset].keys()])
                    break
            
            if base_procs:
                ideal_procs = np.array([1, 2, 4, 8, 16, 32, 64])
                ideal_speedup = ideal_procs / base_procs
                ax.plot(ideal_procs, ideal_speedup, 'k--', label='Идеальное', linewidth=2, alpha=0.5)
            
            for exp_name, data in self.results['experiments'].items():
                if dataset not in data:
                    continue
                
                proc_results = data[dataset]
                procs = sorted([int(p) for p in proc_results.keys()])
                speedups = [proc_results[str(p)].get('speedup', 1.0) for p in procs]
                
                ax.plot(procs, speedups, 'o-', label=exp_name, linewidth=2, markersize=8)
            
            ax.set_xlabel('Количество процессов', fontsize=11)
            ax.set_ylabel('Ускорение', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.set_xscale('log', base=2)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        output_file = self.output_dir / 'strong_scaling_speedup.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ График сохранён: {output_file}")
    
    def plot_efficiency(self):
        """График эффективности"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Эффективность параллелизации', fontsize=14, fontweight='bold')
        
        datasets = ['setA', 'setB', 'setC']
        titles = ['Набор A', 'Набор B', 'Набор C']
        
        for idx, (dataset, title) in enumerate(zip(datasets, titles)):
            ax = axes[idx]
            
            # Идеальная эффективность
            ax.axhline(y=1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Идеальная')
            
            for exp_name, data in self.results['experiments'].items():
                if dataset not in data:
                    continue
                
                proc_results = data[dataset]
                procs = sorted([int(p) for p in proc_results.keys()])
                efficiencies = [proc_results[str(p)].get('efficiency', 1.0) for p in procs]
                
                ax.plot(procs, efficiencies, 'o-', label=exp_name, linewidth=2, markersize=8)
            
            ax.set_xlabel('Количество процессов', fontsize=11)
            ax.set_ylabel('Эффективность', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.set_xscale('log', base=2)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        output_file = self.output_dir / 'efficiency.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ График сохранён: {output_file}")
    
    def plot_comparison_by_dataset(self):
        """Сравнение алгоритмов для каждого набора данных"""
        datasets = ['setA', 'setB', 'setC']
        titles = ['Набор A (N=200, M=20M)', 'Набор B (N=500, M=8M)', 'Набор C (N=1000, M=2M)']
        
        for dataset, title in zip(datasets, titles):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Сравнение алгоритмов: {title}', fontsize=14, fontweight='bold')
            
            # Ускорение
            ax1.set_title('Ускорение', fontsize=12)
            
            # Идеальное
            base_procs = None
            for exp_name, data in self.results['experiments'].items():
                if dataset in data:
                    base_procs = min([int(p) for p in data[dataset].keys()])
                    break
            
            if base_procs:
                ideal_procs = np.array([1, 2, 4, 8, 16, 32, 64])
                ideal_speedup = ideal_procs / base_procs
                ax1.plot(ideal_procs, ideal_speedup, 'k--', label='Идеальное', linewidth=2, alpha=0.5)
            
            for exp_name, data in self.results['experiments'].items():
                if dataset not in data:
                    continue
                
                proc_results = data[dataset]
                procs = sorted([int(p) for p in proc_results.keys()])
                speedups = [proc_results[str(p)].get('speedup', 1.0) for p in procs]
                
                ax1.plot(procs, speedups, 'o-', label=exp_name, linewidth=2, markersize=8)
            
            ax1.set_xlabel('Количество процессов')
            ax1.set_ylabel('Ускорение')
            ax1.set_xscale('log', base=2)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Эффективность
            ax2.set_title('Эффективность', fontsize=12)
            ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Идеальная')
            
            for exp_name, data in self.results['experiments'].items():
                if dataset not in data:
                    continue
                
                proc_results = data[dataset]
                procs = sorted([int(p) for p in proc_results.keys()])
                efficiencies = [proc_results[str(p)].get('efficiency', 1.0) for p in procs]
                
                ax2.plot(procs, efficiencies, 'o-', label=exp_name, linewidth=2, markersize=8)
            
            ax2.set_xlabel('Количество процессов')
            ax2.set_ylabel('Эффективность')
            ax2.set_xscale('log', base=2)
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            output_file = self.output_dir / f'comparison_{dataset}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ График сохранён: {output_file}")
    
    def plot_amdahl_analysis(self):
        """Анализ закона Амдала"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Анализ закона Амдала', fontsize=14, fontweight='bold')
        
        # Теоретические кривые для разных долей последовательного кода
        procs = np.array([1, 2, 4, 8, 16, 32, 64])
        serial_fractions = [0.01, 0.05, 0.10, 0.20]
        
        ax1.set_title('Теоретическое ускорение', fontsize=12)
        for f in serial_fractions:
            speedup = 1 / (f + (1-f)/procs)
            ax1.plot(procs, speedup, 'o-', label=f'f={f*100:.0f}%', linewidth=2, markersize=6)
        
        ax1.plot(procs, procs, 'k--', label='Идеальное', linewidth=2, alpha=0.5)
        ax1.set_xlabel('Количество процессов')
        ax1.set_ylabel('Ускорение')
        ax1.set_xscale('log', base=2)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Фактические данные из экспериментов
        ax2.set_title('Фактическое ускорение (Набор B)', fontsize=12)
        
        dataset = 'setB'
        for exp_name, data in self.results['experiments'].items():
            if dataset not in data:
                continue
            
            proc_results = data[dataset]
            procs_list = sorted([int(p) for p in proc_results.keys()])
            speedups = [proc_results[str(p)].get('speedup', 1.0) for p in procs_list]
            
            ax2.plot(procs_list, speedups, 'o-', label=exp_name, linewidth=2, markersize=8)
        
        # Идеальное
        base_procs = min(procs_list)
        ideal_speedup = np.array(procs_list) / base_procs
        ax2.plot(procs_list, ideal_speedup, 'k--', label='Идеальное', linewidth=2, alpha=0.5)
        
        ax2.set_xlabel('Количество процессов')
        ax2.set_ylabel('Ускорение')
        ax2.set_xscale('log', base=2)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        output_file = self.output_dir / 'amdahl_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ График сохранён: {output_file}")
    
    def generate_all_plots(self):
        """Генерирует все графики"""
        print("\n" + "="*70)
        print("ГЕНЕРАЦИЯ ГРАФИКОВ")
        print("="*70)
        
        self.plot_strong_scaling_time()
        self.plot_strong_scaling_speedup()
        self.plot_efficiency()
        self.plot_comparison_by_dataset()
        self.plot_amdahl_analysis()
        
        print("\n✓ Все графики сгенерированы в директории:", self.output_dir)

if __name__ == "__main__":
    visualizer = PerformanceVisualizer()
    visualizer.generate_all_plots()
