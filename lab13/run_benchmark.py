#!/usr/bin/env python
"""
Скрипт для запуска комплексного бенчмаркинга
базовой и оптимизированной версий CG
"""
import subprocess
import json
import os
import numpy as np

def run_mpi_script(script_name, num_procs, data_prefix):
    """
    Запускает MPI-скрипт
    """
    cmd = f"mpiexec --allow-run-as-root -n {num_procs} python {script_name} {data_prefix}"
    print(f"Запуск: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Ошибка: {result.stderr}")
        return None
    
    print(result.stdout)
    return result.returncode == 0

def collect_results(version, proc_counts, data_prefixes):
    """
    Собирает результаты для анализа
    """
    all_results = {}
    
    for data_prefix in data_prefixes:
        all_results[data_prefix] = {}
        for np in proc_counts:
            result_file = f'../results/profile_{version}_p{np}_{data_prefix}.json'
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    all_results[data_prefix][np] = json.load(f)
    
    return all_results

def main():
    print("="*70)
    print("ЗАПУСК КОМПЛЕКСНОГО БЕНЧМАРКИНГА")
    print("="*70)
    
    # Параметры тестирования
    proc_counts = [1, 2, 4, 8]
    data_prefixes = ["test_500x100", "test_1000x200", "test_2000x400"]
    
    # Запуск базовой версии
    print("\n" + "="*70)
    print("БАЗОВАЯ ВЕРСИЯ")
    print("="*70)
    
    for data_prefix in data_prefixes:
        for np in proc_counts:
            print(f"\nТестирование {data_prefix} с {np} процессами...")
            run_mpi_script("profile_baseline.py", np, data_prefix)
    
    # Запуск оптимизированной версии
    print("\n" + "="*70)
    print("ОПТИМИЗИРОВАННАЯ ВЕРСИЯ")
    print("="*70)
    
    for data_prefix in data_prefixes:
        for np in proc_counts:
            print(f"\nТестирование {data_prefix} с {np} процессами...")
            run_mpi_script("cg_optimized_async.py", np, data_prefix)
    
    # Сбор и сохранение сводных результатов
    print("\n" + "="*70)
    print("СБОР РЕЗУЛЬТАТОВ")
    print("="*70)
    
    baseline_results = collect_results("baseline", proc_counts, data_prefixes)
    optimized_results = collect_results("optimized", proc_counts, data_prefixes)
    
    summary = {
        'baseline': baseline_results,
        'optimized': optimized_results
    }
    
    with open('../results/benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nРезультаты сохранены в ../results/benchmark_summary.json")
    print("\nБенчмаркинг завершён!")

if __name__ == "__main__":
    main()
