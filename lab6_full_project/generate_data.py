#!/usr/bin/env python3
"""Генератор тестовых данных для Lab6"""

import numpy as np

datasets = [
    (100, 100, 'small'),
    (500, 500, 'medium'),
    (1000, 1000, 'large'),
]

print("Генерация тестовых данных для Lab6...")

for M, N, name in datasets:
    np.random.seed(42)
    A = np.random.randn(M, N)
    b = np.random.randn(M)
    x_true = np.random.randn(N)
    
    with open(f'{name}_in.dat', 'w') as f:
        f.write(f"{M} {N}\n")
    
    np.savetxt(f'{name}_A.dat', A, fmt='%.10f')
    np.savetxt(f'{name}_b.dat', b, fmt='%.10f')
    np.savetxt(f'{name}_x_true.dat', x_true, fmt='%.10f')
    
    print(f"✓ {name}: {M}×{N}")

# Копируем small для быстрого тестирования
import shutil
for ext in ['in.dat', 'A.dat', 'b.dat', 'x_true.dat']:
    shutil.copy(f'small_{ext}', ext)

print("\n✓ Данные сгенерированы!")
