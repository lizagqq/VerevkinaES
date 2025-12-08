#!/usr/bin/env python3
"""
Лабораторная работа №6, Часть 1: Базовые операции с декартовой топологией
Создание топологии "тор", определение соседей, кольцевой обмен
"""

from mpi4py import MPI
import numpy as np
import sys

def main():
    # Инициализация MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Проверка: размер должен быть квадратом
    sqrt_size = int(np.sqrt(size))
    if sqrt_size * sqrt_size != size:
        if rank == 0:
            print(f"ОШИБКА: Количество процессов ({size}) должно быть полным квадратом!")
            print(f"Допустимые значения: 1, 4, 9, 16, 25, 36, 49, 64, ...")
        sys.exit(1)
    
    num_row = num_col = sqrt_size
    
    if rank == 0:
        print("="*70)
        print("ЧАСТЬ 1: БАЗОВЫЕ ОПЕРАЦИИ С ДЕКАРТОВОЙ ТОПОЛОГИЕЙ")
        print("="*70)
        print(f"Сетка процессов: {num_row} × {num_col} = {size} процессов")
        print(f"Топология: ТОР (периодические границы)\n")
    
    # Этап 1.1: Создание декартовой топологии типа "тор"
    dims = [num_row, num_col]
    periods = [True, True]  # Периодические границы (тор)
    reorder = True  # Разрешаем MPI переупорядочивать ранги
    
    comm_cart = comm.Create_cart(dims=dims, periods=periods, reorder=reorder)
    cart_rank = comm_cart.Get_rank()
    coords = comm_cart.Get_coords(cart_rank)
    
    # Вывод информации о новых рангах
    print(f"Процесс old_rank={rank} → cart_rank={cart_rank}, "
          f"координаты=({coords[0]}, {coords[1]})")
    comm.Barrier()
    
    # Этап 1.2: Определение соседей
    if rank == 0:
        print("\n" + "="*70)
        print("ОПРЕДЕЛЕНИЕ СОСЕДЕЙ")
        print("="*70)
    comm.Barrier()
    
    # Направление 0 (вертикаль): вверх/вниз
    neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)
    
    # Направление 1 (горизонталь): влево/вправо
    neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)
    
    print(f"Процесс cart_rank={cart_rank} ({coords[0]},{coords[1]}): "
          f"↑{neighbour_up} ↓{neighbour_down} ←{neighbour_left} →{neighbour_right}")
    comm.Barrier()
    
    # Этап 1.3: Кольцевой обмен с Sendrecv_replace
    if rank == 0:
        print("\n" + "="*70)
        print("КОЛЬЦЕВОЙ ОБМЕН (ГОРИЗОНТАЛЬНОЕ КОЛЬЦО)")
        print("="*70)
    
    # Создаём массив с уникальными данными на каждом процессе
    a = np.array([cart_rank + 1], dtype=np.float64)
    initial_value = a[0]
    
    if rank == 0:
        print(f"Начальные значения: процесс i имеет a[i] = i+1")
    comm.Barrier()
    
    # Кольцевой обмен по горизонтали (вдоль строки)
    sum_value = a[0]
    
    for step in range(num_col):
        # Отправляем направо, принимаем слева
        comm_cart.Sendrecv_replace(
            a, 
            dest=neighbour_right, 
            sendtag=0,
            source=neighbour_left, 
            recvtag=0
        )
        sum_value += a[0]
    
    # После полного обхода каждый процесс должен иметь сумму всех в строке
    expected_sum = sum(range(coords[0] * num_col + 1, (coords[0] + 1) * num_col + 1))
    
    print(f"Процесс {cart_rank}: начальное={initial_value:.0f}, "
          f"сумма после обмена={sum_value:.0f}, ожидаемая={expected_sum}")
    
    comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*70)
        print("ПРОВЕРКА КОРРЕКТНОСТИ")
        print("="*70)
    
    # Проверка корректности
    is_correct = (abs(sum_value - expected_sum) < 1e-10)
    all_correct = comm.allreduce(is_correct, op=MPI.LAND)
    
    if rank == 0:
        if all_correct:
            print("✓ Кольцевой обмен выполнен КОРРЕКТНО!")
        else:
            print("✗ Обнаружены ошибки в кольцевом обмене!")
    
    # Освобождение ресурсов
    comm_cart.Free()

if __name__ == "__main__":
    main()
