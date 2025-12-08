#!/usr/bin/env python3
"""
Скрипт для сбора метрик производительности параллельных программ
Измеряет время выполнения, вычисляет ускорение и эффективность
"""

import json
import sys
import os
from datetime import datetime

class PerformanceCollector:
    """Класс для сбора и анализа метрик производительности"""
    
    def __init__(self, output_file='performance_results.json'):
        self.results = {
            'metadata': {
                'date': datetime.now().isoformat(),
                'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
            },
            'experiments': {}
        }
        self.output_file = output_file
    
    def add_experiment(self, experiment_name, dataset_name, num_procs, 
                      execution_time, iterations=None, residual=None, **kwargs):
        """
        Добавляет результат эксперимента
        
        Parameters:
        experiment_name - имя эксперимента (например, 'cg_simple')
        dataset_name - имя набора данных ('setA', 'setB', etc.)
        num_procs - количество процессов
        execution_time - время выполнения в секундах
        iterations - количество итераций (для итеративных методов)
        residual - финальная невязка
        **kwargs - дополнительные метрики
        """
        if experiment_name not in self.results['experiments']:
            self.results['experiments'][experiment_name] = {}
        
        if dataset_name not in self.results['experiments'][experiment_name]:
            self.results['experiments'][experiment_name][dataset_name] = {}
        
        result_entry = {
            'time': execution_time,
            'processes': num_procs
        }
        
        if iterations is not None:
            result_entry['iterations'] = iterations
        if residual is not None:
            result_entry['residual'] = residual
        
        result_entry.update(kwargs)
        
        self.results['experiments'][experiment_name][dataset_name][str(num_procs)] = result_entry
        
        print(f"✓ Записан результат: {experiment_name}/{dataset_name}/P={num_procs}: "
              f"{execution_time:.6f}s")
    
    def calculate_metrics(self):
        """Вычисляет ускорение и эффективность для всех экспериментов"""
        print("\nВычисление метрик производительности...")
        
        for exp_name, datasets in self.results['experiments'].items():
            for dataset_name, proc_results in datasets.items():
                # Находим базовое время (наименьшее количество процессов)
                proc_counts = sorted([int(p) for p in proc_results.keys()])
                
                if not proc_counts:
                    continue
                
                base_procs = proc_counts[0]
                base_time = proc_results[str(base_procs)]['time']
                
                # Вычисляем метрики для каждого количества процессов
                for num_procs in proc_counts:
                    current_time = proc_results[str(num_procs)]['time']
                    
                    # Ускорение (Speedup)
                    speedup = base_time / current_time if current_time > 0 else 0
                    
                    # Эффективность (Efficiency)
                    # E = S / P, где P - отношение процессов к базовому
                    proc_ratio = num_procs / base_procs
                    efficiency = speedup / proc_ratio if proc_ratio > 0 else 0
                    
                    # Обновляем результаты
                    proc_results[str(num_procs)]['speedup'] = speedup
                    proc_results[str(num_procs)]['efficiency'] = efficiency
                    proc_results[str(num_procs)]['base_time'] = base_time
                    proc_results[str(num_procs)]['base_procs'] = base_procs
                    
                print(f"  ✓ {exp_name}/{dataset_name}: метрики для {len(proc_counts)} конфигураций")
    
    def save(self):
        """Сохраняет результаты в JSON файл"""
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Результаты сохранены в {self.output_file}")
    
    def load(self, filename=None):
        """Загружает результаты из JSON файла"""
        if filename is None:
            filename = self.output_file
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.results = json.load(f)
            print(f"✓ Результаты загружены из {filename}")
            return True
        return False
    
    def print_summary(self):
        """Выводит сводку результатов"""
        print("\n" + "="*70)
        print("СВОДКА РЕЗУЛЬТАТОВ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*70)
        
        for exp_name, datasets in self.results['experiments'].items():
            print(f"\n{exp_name.upper()}:")
            print("-" * 70)
            
            for dataset_name, proc_results in datasets.items():
                print(f"\n  Набор данных: {dataset_name}")
                print(f"  {'Процессы':<10} {'Время(с)':<12} {'Ускорение':<12} {'Эффективность':<15}")
                print(f"  {'-'*60}")
                
                proc_counts = sorted([int(p) for p in proc_results.keys()])
                for num_procs in proc_counts:
                    r = proc_results[str(num_procs)]
                    speedup = r.get('speedup', 1.0)
                    efficiency = r.get('efficiency', 1.0)
                    print(f"  {num_procs:<10} {r['time']:<12.6f} {speedup:<12.2f} {efficiency*100:<15.1f}%")

def create_synthetic_results():
    """
    Создаёт синтетические результаты для демонстрации
    Основано на реалистичных паттернах производительности
    """
    collector = PerformanceCollector('synthetic_results.json')
    
    print("="*70)
    print("ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ РЕЗУЛЬТАТОВ")
    print("="*70)
    print("\nМоделирование результатов бенчмарков...")
    print("(В реальной работе эти данные получаются из фактических запусков)")
    
    # Параметры для моделирования
    datasets = {
        'setA': {'base_time': 45.2, 'parallel_fraction': 0.92, 'M': 20_000_000, 'N': 200},
        'setB': {'base_time': 32.8, 'parallel_fraction': 0.94, 'M': 8_000_000, 'N': 500},
        'setC': {'base_time': 21.5, 'parallel_fraction': 0.96, 'M': 2_000_000, 'N': 1000},
    }
    
    experiments = ['cg_simple', 'cg_full', 'matvec']
    process_counts = [1, 2, 4, 8, 16, 32, 64]
    
    # Закон Амдала: T(p) = T_seq * (f + (1-f)/p)
    # где f - последовательная доля, (1-f) - параллельная доля
    
    for exp_name in experiments:
        print(f"\nГенерация для {exp_name}...")
        
        # Модификаторы для разных экспериментов
        if exp_name == 'cg_simple':
            efficiency_factor = 1.0  # Лучшая эффективность
        elif exp_name == 'cg_full':
            efficiency_factor = 0.85  # Больше коммуникаций
        else:  # matvec
            efficiency_factor = 0.90
        
        for dataset_name, params in datasets.items():
            base_time = params['base_time']
            f = 1 - params['parallel_fraction']  # Последовательная доля
            
            for num_procs in process_counts:
                # Закон Амдала + накладные расходы на коммуникацию
                comm_overhead = 0.001 * num_procs * (num_procs - 1)  # Квадратичный рост
                
                # Время выполнения
                amdahl_time = base_time * (f + (1 - f) / num_procs)
                total_time = (amdahl_time + comm_overhead) / efficiency_factor
                
                # Добавляем небольшой шум (±2%)
                import random
                noise = 1.0 + random.uniform(-0.02, 0.02)
                total_time *= noise
                
                # Для итеративных методов
                iterations = 500 if 'cg' in exp_name else None
                residual = 1e-10 if 'cg' in exp_name else None
                
                collector.add_experiment(
                    exp_name, dataset_name, num_procs,
                    total_time, iterations, residual
                )
    
    collector.calculate_metrics()
    collector.print_summary()
    collector.save()
    
    return collector

if __name__ == "__main__":
    # Создаём синтетические результаты для демонстрации
    collector = create_synthetic_results()
    
    print("\n" + "="*70)
    print("СИНТЕТИЧЕСКИЕ ДАННЫЕ СОЗДАНЫ")
    print("="*70)
    print("\nВ реальной работе:")
    print("  1. Запустите программы на разных количествах процессов")
    print("  2. Соберите время выполнения с помощью MPI.Wtime()")
    print("  3. Используйте PerformanceCollector для записи результатов")
    print("  4. Вызовите calculate_metrics() и save()")
