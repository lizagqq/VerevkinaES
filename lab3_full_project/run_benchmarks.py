import subprocess
import json
import os
import numpy as np

def run_command(cmd):
    """Выполняет команду и возвращает вывод"""
    env = os.environ.copy()
    env['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        shell=False
    )
    return result.stdout, result.stderr

def extract_time(output):
    """Извлекает время выполнения из вывода"""
    for line in output.split('\n'):
        if 'Время выполнения:' in line:
            return float(line.split(':')[1].strip().split()[0])
    return None

def extract_iterations(output):
    """Извлекает количество итераций из вывода"""
    for line in output.split('\n'):
        if 'Количество итераций:' in line:
            return int(line.split(':')[1].strip())
    return None

def extract_residual(output):
    """Извлекает финальную невязку из вывода"""
    for line in output.split('\n'):
        if 'Финальная норма невязки:' in line:
            return float(line.split(':')[1].strip())
    return None

def benchmark_version(script_name, process_counts, system_prefix=''):
    """
    Бенчмарк для одной версии на разном количестве процессов
    """
    results = {}
    
    for np_count in process_counts:
        print(f"\n  Запуск с {np_count} процессами...")
        
        # Копируем файлы данных
        if system_prefix:
            os.system(f'cp {system_prefix}in.dat in.dat')
            os.system(f'cp {system_prefix}AData.dat AData.dat')
            os.system(f'cp {system_prefix}bData.dat bData.dat')
        
        stdout, stderr = run_command([
            'mpiexec', '-n', str(np_count),
            'python', script_name
        ])
        
        exec_time = extract_time(stdout)
        iterations = extract_iterations(stdout)
        residual = extract_residual(stdout)
        
        if exec_time:
            print(f"    Время: {exec_time:.6f} сек, Итерации: {iterations}, Невязка: {residual:.2e}")
            
            results[np_count] = {
                'time': exec_time,
                'iterations': iterations,
                'residual': residual
            }
    
    # Вычисляем ускорение и эффективность
    if 1 in results:
        seq_time = results[1]['time']
        for np_count in process_counts:
            if np_count > 1 and np_count in results:
                speedup = seq_time / results[np_count]['time']
                efficiency = speedup / np_count
                results[np_count]['speedup'] = speedup
                results[np_count]['efficiency'] = efficiency
    
    return results

def main():
    print("=" * 70)
    print("БЕНЧМАРК МЕТОДА СОПРЯЖЁННЫХ ГРАДИЕНТОВ")
    print("=" * 70)
    
    process_counts = [1, 2, 4]
    
    # Маленькая система для быстрого тестирования
    system_sizes = {
        'small': ('small_', "Маленькая система (20x10)"),
        'medium': ('medium_', "Средняя система (200x100)"),
        'main': ('', "Основная система (1000x500)")
    }
    
    all_results = {}
    
    for size_key, (prefix, description) in system_sizes.items():
        print(f"\n{'='*70}")
        print(f"{description}")
        print(f"{'='*70}")
        
        all_results[size_key] = {}
        
        # Бенчмарк полной версии
        print("\n1. Полная версия (cg_full.py):")
        print("-" * 70)
        results_full = benchmark_version('cg_full.py', process_counts, prefix)
        all_results[size_key]['full'] = results_full
        
        # Бенчмарк упрощённой версии
        print("\n2. Упрощённая версия (cg_simple.py):")
        print("-" * 70)
        results_simple = benchmark_version('cg_simple.py', process_counts, prefix)
        all_results[size_key]['simple'] = results_simple
    
    # Сохранение результатов
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 70)
    
    for size_key, (prefix, description) in system_sizes.items():
        print(f"\n{description}")
        print("-" * 70)
        
        if size_key in all_results:
            data = all_results[size_key]
            
            print("\nПолная версия:")
            print(f"{'Процессы':<10} {'Время (с)':<12} {'Итерации':<12} {'Ускорение':<12} {'Эффективность':<15}")
            print("-" * 70)
            for np_count in sorted(data['full'].keys()):
                r = data['full'][np_count]
                speedup = r.get('speedup', 1.0)
                efficiency = r.get('efficiency', 1.0)
                print(f"{np_count:<10} {r['time']:<12.6f} {r['iterations']:<12} "
                      f"{speedup:<12.2f} {efficiency*100:<15.1f}%")
            
            print("\nУпрощённая версия:")
            print(f"{'Процессы':<10} {'Время (с)':<12} {'Итерации':<12} {'Ускорение':<12} {'Эффективность':<15}")
            print("-" * 70)
            for np_count in sorted(data['simple'].keys()):
                r = data['simple'][np_count]
                speedup = r.get('speedup', 1.0)
                efficiency = r.get('efficiency', 1.0)
                print(f"{np_count:<10} {r['time']:<12.6f} {r['iterations']:<12} "
                      f"{speedup:<12.2f} {efficiency*100:<15.1f}%")
            
            # Сравнение версий
            print("\nСравнение полной и упрощённой версий:")
            print(f"{'Процессы':<10} {'Полная (с)':<15} {'Упрощ. (с)':<15} {'Разница':<15}")
            print("-" * 70)
            for np_count in sorted(data['full'].keys()):
                if np_count in data['simple']:
                    t_full = data['full'][np_count]['time']
                    t_simple = data['simple'][np_count]['time']
                    ratio = t_full / t_simple if t_simple > 0 else 0
                    print(f"{np_count:<10} {t_full:<15.6f} {t_simple:<15.6f} {ratio:<15.2f}x")
    
    print("\n" + "=" * 70)
    print("Результаты сохранены в benchmark_results.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
