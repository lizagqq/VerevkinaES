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

def extract_times(output):
    """Извлекает времена выполнения из вывода"""
    par_time = None
    seq_time = None
    speedup = None
    
    for line in output.split('\n'):
        if 'Время параллельное:' in line or 'Время параллельного выполнения:' in line:
            par_time = float(line.split(':')[1].strip().split()[0])
        elif 'Время последовательное:' in line or 'Время последовательного выполнения:' in line:
            seq_time = float(line.split(':')[1].strip().split()[0])
        elif 'Ускорение:' in line:
            speedup_str = line.split(':')[1].strip().split('x')[0]
            try:
                speedup = float(speedup_str)
            except:
                pass
    
    return par_time, seq_time, speedup

def benchmark_part1():
    """Бенчмарк для Части 1"""
    print("\n" + "=" * 60)
    print("БЕНЧМАРК ЧАСТИ 1: СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ")
    print("=" * 60)
    
    from generate_data import generate_test_data_part1
    
    results = {}
    vector_sizes = [1000, 5000, 10000]
    process_counts = [1, 2, 4]
    
    for M in vector_sizes:
        print(f"\n{'='*60}")
        print(f"Размер вектора: {M}")
        print(f"{'='*60}")
        
        generate_test_data_part1(M, seed=42)
        
        size_key = f"M={M}"
        results[size_key] = {}
        
        for np_count in process_counts:
            print(f"\nЗапуск с {np_count} процессами...")
            
            if np_count == 1:
                # Для 1 процесса запускаем специальную версию
                stdout, _ = run_command([
                    'mpiexec', '-n', '1',
                    'python', 'lab2_part1_reduce.py'
                ])
            else:
                stdout, _ = run_command([
                    'mpiexec', '-n', str(np_count),
                    'python', 'lab2_part1_reduce.py'
                ])
            
            par_time, seq_time, speedup = extract_times(stdout)
            
            if par_time and seq_time:
                efficiency = speedup / np_count if speedup else 0
                
                print(f"  Время параллельное: {par_time:.6f} сек")
                print(f"  Время последовательное: {seq_time:.6f} сек")
                print(f"  Ускорение: {speedup:.2f}x")
                print(f"  Эффективность: {efficiency:.2%}")
                
                results[size_key][np_count] = {
                    'parallel_time': par_time,
                    'sequential_time': seq_time,
                    'speedup': speedup,
                    'efficiency': efficiency
                }
    
    return results

def benchmark_part2():
    """Бенчмарк для Части 2"""
    print("\n" + "=" * 60)
    print("БЕНЧМАРК ЧАСТИ 2: УМНОЖЕНИЕ A.T @ x")
    print("=" * 60)
    
    from generate_data import generate_test_data_part2
    
    results = {}
    matrix_sizes = [(100, 80), (500, 400), (1000, 800)]
    process_counts = [1, 2, 4]
    
    for M, N in matrix_sizes:
        print(f"\n{'='*60}")
        print(f"Размер матрицы: {M} x {N}")
        print(f"{'='*60}")
        
        generate_test_data_part2(M, N, seed=42)
        
        size_key = f"{M}x{N}"
        results[size_key] = {}
        
        for np_count in process_counts:
            print(f"\nЗапуск с {np_count} процессами...")
            
            stdout, _ = run_command([
                'mpiexec', '-n', str(np_count),
                'python', 'lab2_part2.py'
            ])
            
            par_time, seq_time, speedup = extract_times(stdout)
            
            if par_time and seq_time:
                efficiency = speedup / np_count if speedup else 0
                
                print(f"  Время параллельное: {par_time:.6f} сек")
                print(f"  Время последовательное: {seq_time:.6f} сек")
                print(f"  Ускорение: {speedup:.2f}x")
                print(f"  Эффективность: {efficiency:.2%}")
                
                results[size_key][np_count] = {
                    'parallel_time': par_time,
                    'sequential_time': seq_time,
                    'speedup': speedup,
                    'efficiency': efficiency
                }
    
    return results

def main():
    print("=" * 60)
    print("БЕНЧМАРК ЛАБОРАТОРНОЙ РАБОТЫ №2")
    print("=" * 60)
    
    all_results = {
        'part1': benchmark_part1(),
        'part2': benchmark_part2()
    }
    
    # Сохранение результатов
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Результаты сохранены в benchmark_results.json")
    print("=" * 60)
    
    # Вывод сводной таблицы
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    print("\nЧАСТЬ 1: СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ")
    print("-" * 80)
    for size, data in all_results['part1'].items():
        print(f"\n{size}")
        print(f"{'Процессы':<12} {'Время пар.(с)':<15} {'Время посл.(с)':<15} {'Ускорение':<12} {'Эффективность':<15}")
        print("-" * 80)
        for proc, values in sorted(data.items()):
            print(f"{proc:<12} {values['parallel_time']:<15.6f} {values['sequential_time']:<15.6f} "
                  f"{values['speedup']:<12.2f} {values['efficiency']*100:<15.1f}%")
    
    print("\n\nЧАСТЬ 2: УМНОЖЕНИЕ A.T @ x")
    print("-" * 80)
    for size, data in all_results['part2'].items():
        print(f"\n{size}")
        print(f"{'Процессы':<12} {'Время пар.(с)':<15} {'Время посл.(с)':<15} {'Ускорение':<12} {'Эффективность':<15}")
        print("-" * 80)
        for proc, values in sorted(data.items()):
            print(f"{proc:<12} {values['parallel_time']:<15.6f} {values['sequential_time']:<15.6f} "
                  f"{values['speedup']:<12.2f} {values['efficiency']*100:<15.1f}%")

if __name__ == "__main__":
    main()
