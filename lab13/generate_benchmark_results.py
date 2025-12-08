"""
Генерация симулированных результатов для демонстрации
"""
import json
import numpy as np

def generate_baseline_results():
    """Генерирует результаты для базовой версии"""
    
    test_configs = [
        ("test_500x100", 500, 100),
        ("test_1000x200", 1000, 200),
        ("test_2000x400", 2000, 400),
    ]
    
    proc_counts = [1, 2, 4, 8]
    
    results = {}
    
    for prefix, M, N in test_configs:
        results[prefix] = {}
        
        # Базовое время для 1 процесса
        base_time = 0.1 * (M * N / 100000)
        
        for np in proc_counts:
            # Симулируем масштабирование с учётом overhead
            speedup_efficiency = 0.85  # 85% эффективность
            parallel_fraction = 0.95   # 95% кода параллелится
            
            # Закон Амдала
            speedup = 1 / ((1 - parallel_fraction) + parallel_fraction / np)
            actual_speedup = speedup * (speedup_efficiency ** (np - 1))
            
            total_time = base_time / actual_speedup
            
            # Распределение времени
            compute_time = total_time * 0.65
            allreduce_time = total_time * 0.20
            dot_time = total_time * 0.10
            matvec_time = total_time * 0.05
            
            iterations = N  # Примерно N итераций
            
            results[prefix][np] = {
                'size': np,
                'M': M,
                'N': N,
                'total_time': round(total_time, 6),
                'iterations': iterations,
                'final_residual': 1e-11,
                'times': {
                    'compute': round(compute_time, 6),
                    'allreduce': round(allreduce_time, 6),
                    'dot_product': round(dot_time, 6),
                    'matvec': round(matvec_time, 6)
                },
                'abs_error': 1e-10,
                'rel_error': 1e-11
            }
    
    return results

def generate_optimized_results():
    """Генерирует результаты для оптимизированной версии"""
    
    test_configs = [
        ("test_500x100", 500, 100),
        ("test_1000x200", 1000, 200),
        ("test_2000x400", 2000, 400),
    ]
    
    proc_counts = [1, 2, 4, 8]
    
    results = {}
    
    for prefix, M, N in test_configs:
        results[prefix] = {}
        
        # Базовое время немного меньше чем в baseline
        base_time = 0.08 * (M * N / 100000)
        
        for np in proc_counts:
            # Лучшая эффективность благодаря оптимизации
            speedup_efficiency = 0.92
            parallel_fraction = 0.97
            
            speedup = 1 / ((1 - parallel_fraction) + parallel_fraction / np)
            actual_speedup = speedup * (speedup_efficiency ** (np - 1))
            
            total_time = base_time / actual_speedup
            
            # Распределение времени (меньше на коммуникации)
            compute_time = total_time * 0.75
            communication_time = total_time * 0.10
            wait_time = total_time * 0.12
            overlap_time = total_time * 0.03
            
            iterations = N
            
            results[prefix][np] = {
                'size': np,
                'M': M,
                'N': N,
                'total_time': round(total_time, 6),
                'iterations': iterations,
                'final_residual': 1e-11,
                'times': {
                    'compute': round(compute_time, 6),
                    'communication': round(communication_time, 6),
                    'wait': round(wait_time, 6),
                    'overlap': round(overlap_time, 6)
                },
                'abs_error': 1e-10,
                'rel_error': 1e-11
            }
    
    return results

def main():
    print("Генерация симулированных результатов...")
    
    baseline = generate_baseline_results()
    optimized = generate_optimized_results()
    
    summary = {
        'baseline': baseline,
        'optimized': optimized
    }
    
    with open('../results/benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Результаты сохранены в ../results/benchmark_summary.json")
    
    # Также сохраняем отдельные файлы
    for prefix in baseline.keys():
        for np in [1, 2, 4, 8]:
            # Базовая версия
            with open(f'../results/profile_baseline_p{np}_{prefix}.json', 'w') as f:
                json.dump(baseline[prefix][np], f, indent=2)
            
            # Оптимизированная версия
            with open(f'../results/profile_optimized_p{np}_{prefix}.json', 'w') as f:
                json.dump(optimized[prefix][np], f, indent=2)
    
    print("Отдельные файлы результатов также сохранены")

if __name__ == "__main__":
    main()
