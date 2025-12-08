import subprocess
import numpy as np
import time
import json

def run_benchmark(script_name, num_processes, iterations=3):
    """
    Запускает скрипт несколько раз и возвращает среднее время выполнения
    """
    times = []
    
    for i in range(iterations):
        if script_name.endswith('sequential.py'):
            # Последовательная версия
            result = subprocess.run(
                ['python', script_name],
                capture_output=True,
                text=True
            )
        else:
            # Параллельные версии
            result = subprocess.run(
                ['mpiexec', '-n', str(num_processes), 'python', script_name],
                capture_output=True,
                text=True
            )
        
        # Извлекаем время из вывода
        for line in result.stdout.split('\n'):
            if 'Время выполнения:' in line:
                exec_time = float(line.split(':')[1].strip().split()[0])
                times.append(exec_time)
                break
    
    if times:
        return np.mean(times), np.std(times)
    else:
        return None, None

def verify_results():
    """
    Проверяет корректность всех параллельных версий
    """
    print("=" * 60)
    print("ВЕРИФИКАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    # Загружаем эталонный результат
    b_seq = np.loadtxt('Results_sequential.dat')
    
    # Проверяем результаты параллельных версий
    results = {
        'Send/Recv': 'Results_parallel_sendrecv.dat',
        'Collective': 'Results_parallel_collective.dat',
        'Arbitrary': 'Results_parallel_arbitrary.dat'
    }
    
    for name, filename in results.items():
        try:
            b_par = np.loadtxt(filename)
            diff = np.max(np.abs(b_seq - b_par))
            print(f"{name:15s}: Макс. отклонение = {diff:.2e}", end="")
            if diff < 1e-10:
                print(" ✓ КОРРЕКТНО")
            else:
                print(" ✗ ОШИБКА")
        except FileNotFoundError:
            print(f"{name:15s}: Файл не найден")

def main():
    """
    Основная функция для проведения бенчмарков
    """
    print("=" * 60)
    print("БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    # Конфигурация для тестирования
    matrix_sizes = [
        (100, 100),
        (500, 500),
        (1000, 1000)
    ]
    
    process_counts = [1, 2, 4, 8]
    
    results = {}
    
    for M, N in matrix_sizes:
        print(f"\n{'=' * 60}")
        print(f"Размер матрицы: {M} x {N}")
        print(f"{'=' * 60}")
        
        # Генерируем тестовые данные
        from generate_data import generate_test_data
        generate_test_data(M, N)
        
        # Запускаем последовательную версию
        print("\nПоследовательная версия...")
        subprocess.run(['python', 'lab1_sequential.py'], capture_output=True)
        seq_time, seq_std = run_benchmark('lab1_sequential.py', 1)
        print(f"  Время: {seq_time:.6f} ± {seq_std:.6f} сек")
        
        size_key = f"{M}x{N}"
        results[size_key] = {
            'sequential': seq_time,
            'parallel': {}
        }
        
        # Запускаем параллельные версии
        for np_count in process_counts:
            if np_count == 1:
                continue
                
            print(f"\nПараллельная версия (произвольный размер), {np_count} процессов...")
            
            # Запускаем версию с произвольным размером (самая универсальная)
            mean_time, std_time = run_benchmark('lab1_parallel_arbitrary.py', np_count)
            
            if mean_time:
                speedup = seq_time / mean_time
                efficiency = speedup / np_count
                
                print(f"  Время: {mean_time:.6f} ± {std_time:.6f} сек")
                print(f"  Ускорение: {speedup:.2f}x")
                print(f"  Эффективность: {efficiency:.2%}")
                
                results[size_key]['parallel'][np_count] = {
                    'time': mean_time,
                    'speedup': speedup,
                    'efficiency': efficiency
                }
        
        # Верификация результатов
        verify_results()
    
    # Сохраняем результаты в JSON
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Результаты бенчмарков сохранены в benchmark_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
