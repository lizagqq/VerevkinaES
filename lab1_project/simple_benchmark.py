import subprocess
import numpy as np
import json
import os

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

def main():
    print("=" * 60)
    print("БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    # Создаём результаты для разных размеров
    results = {}
    
    # Размеры для тестирования
    test_cases = [
        (100, 100),
        (500, 500),
        (1000, 1000)
    ]
    
    process_counts = [1, 2, 4, 8]
    
    for M, N in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Тестирование для размера {M}x{N}")
        print(f"{'=' * 60}")
        
        # Генерируем данные
        from generate_data import generate_test_data
        generate_test_data(M, N, seed=42)
        
        # Последовательная версия
        print(f"\nПоследовательная версия...")
        stdout, _ = run_command(['python', 'lab1_sequential.py'])
        seq_time = extract_time(stdout)
        print(f"  Время: {seq_time:.6f} сек")
        
        size_key = f"{M}x{N}"
        results[size_key] = {
            'sequential': seq_time,
            'parallel': {}
        }
        
        # Параллельные версии
        for np_count in process_counts:
            if np_count == 1:
                continue
            
            print(f"\nПараллельная версия, {np_count} процессов...")
            stdout, _ = run_command([
                'mpiexec', '-n', str(np_count),
                'python', 'lab1_parallel_arbitrary.py'
            ])
            
            par_time = extract_time(stdout)
            
            if par_time:
                speedup = seq_time / par_time
                efficiency = speedup / np_count
                
                print(f"  Время: {par_time:.6f} сек")
                print(f"  Ускорение: {speedup:.2f}x")
                print(f"  Эффективность: {efficiency:.2%}")
                
                results[size_key]['parallel'][np_count] = {
                    'time': par_time,
                    'speedup': speedup,
                    'efficiency': efficiency
                }
    
    # Сохраняем результаты
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Результаты сохранены в benchmark_results.json")
    print("=" * 60)
    
    # Выводим сводную таблицу
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    for size, data in results.items():
        print(f"\nРазмер матрицы: {size}")
        print("-" * 80)
        print(f"{'Процессы':<12} {'Время (сек)':<15} {'Ускорение':<15} {'Эффективность':<15}")
        print("-" * 80)
        
        seq_time = data['sequential']
        print(f"{'1 (seq)':<12} {seq_time:<15.6f} {'1.00':<15} {'100.0%':<15}")
        
        for proc_count in sorted([int(k) for k in data['parallel'].keys()]):
            par_data = data['parallel'][str(proc_count)]
            print(f"{proc_count:<12} {par_data['time']:<15.6f} "
                  f"{par_data['speedup']:<15.2f} {par_data['efficiency']*100:<15.1f}%")

if __name__ == "__main__":
    main()
