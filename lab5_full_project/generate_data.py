#!/usr/bin/env python3
"""
Генератор тестовых данных для Lab5
"""

import numpy as np

def generate_test_data(M, N, seed=42):
    """Генерирует тестовые данные для умножения матрицы на вектор"""
    np.random.seed(seed)
    
    A = np.random.randn(M, N)
    x = np.random.randn(N)
    
    # Истинный результат
    b_true = np.dot(A, x)
    
    return A, x, b_true

if __name__ == "__main__":
    # Наборы данных
    datasets = [
        (100, 100, 'small_'),
        (500, 500, 'medium_'),
        (1000, 1000, 'large_'),
    ]
    
    print("="*70)
    print("ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ ДЛЯ LAB5")
    print("="*70)
    
    for M, N, prefix in datasets:
        print(f"\n{prefix[:-1].capitalize()}: {M} × {N}")
        
        A, x, b_true = generate_test_data(M, N)
        
        # Сохранение
        with open(f'{prefix}in.dat', 'w') as f:
            f.write(f"{M} {N}\n")
        
        np.savetxt(f'{prefix}AData.dat', A, fmt='%.10f')
        np.savetxt(f'{prefix}xData.dat', x, fmt='%.10f')
        np.savetxt(f'{prefix}bData_true.dat', b_true, fmt='%.10f')
        
        print(f"  Файлы сохранены с префиксом '{prefix}'")
    
    print("\n✓ Генерация завершена!")
