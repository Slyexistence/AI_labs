import numpy as np

def simplex_max(c, A, b):
    """
    Реализация симплекс-метода для максимизации целевой функции.
    """
    
    try:
        # Преобразование входных данных в numpy массивы
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        
        m, n = A.shape  # m - количество ограничений, n - количество переменных
        
        # Проверка размерностей
        if len(c) != n:
            raise ValueError("Размерность c не соответствует количеству переменных в A")
        if len(b) != m:
            raise ValueError("Размерность b не соответствует количеству ограничений")
        
        # Создаем симплекс-таблицу
        # Добавляем slack переменные (по одной на каждое ограничение ≤)
        tableau = np.zeros((m + 1, n + m + 1))
        
        # Заполняем коэффициенты ограничений и slack переменные
        tableau[:m, :n] = A
        for i in range(m):
            tableau[i, n + i] = 1  # slack переменные
            tableau[i, -1] = b[i]   # правая часть
        
        # Целевая функция (для максимизации берем с минусом в последней строке)
        tableau[m, :n] = -c
        tableau[m, -1] = 0  # значение целевой функции
        
        # Индексы базисных переменных
        basis = list(range(n, n + m))
        
        iteration = 0
        max_iterations = 1000  # защита от зацикливания
        
        print("Начальная симплекс-таблица:")
        print_tableau(tableau, basis, n)
        
        while iteration < max_iterations:
            iteration += 1
            
            # Проверка на оптимальность (все коэффициенты в последней строке >= 0)
            if np.all(tableau[-1, :-1] >= -1e-10):  # допускаем небольшую погрешность
                break
            
            # Выбор ведущего столбца (минимальный отрицательный коэффициент)
            pivot_col = np.argmin(tableau[-1, :-1])
            
            # Проверка на неограниченность решения
            if np.all(tableau[:-1, pivot_col] <= 0):
                return {
                    'solution': None,
                    'max_value': float('inf'),
                    'iterations': iteration,
                    'success': False,
                    'message': 'Задача не ограничена (решение уходит в бесконечность)'
                }
            
            # Выбор ведущей строки (минимальное положительное отношение b/коэфф)
            ratios = []
            for i in range(m):
                if tableau[i, pivot_col] > 1e-10:  # положительный коэффициент
                    ratio = tableau[i, -1] / tableau[i, pivot_col]
                    ratios.append((ratio, i))
            
            if not ratios:
                return {
                    'solution': None,
                    'max_value': float('inf'),
                    'iterations': iteration,
                    'success': False,
                    'message': 'Задача не ограничена (нет положительных отношений)'
                }
            
            # Выбираем строку с минимальным отношением
            pivot_row = min(ratios)[1]
            
            # Запоминаем ведущий элемент
            pivot_element = tableau[pivot_row, pivot_col]
            
            print(f"\nИтерация {iteration}:")
            print(f"Ведущий элемент: строка {pivot_row + 1}, столбец {pivot_col + 1}, значение = {pivot_element:.3f}")
            
            # Обновляем базис
            basis[pivot_row] = pivot_col
            
            # Делим ведущую строку на ведущий элемент
            tableau[pivot_row, :] /= pivot_element
            
            # Обнуляем остальные строки в ведущем столбце
            for i in range(m + 1):
                if i != pivot_row:
                    factor = tableau[i, pivot_col]
                    if abs(factor) > 1e-10:
                        tableau[i, :] -= factor * tableau[pivot_row, :]
            
            print_tableau(tableau, basis, n)
        
        if iteration >= max_iterations:
            return {
                'solution': None,
                'max_value': None,
                'iterations': iteration,
                'success': False,
                'message': 'Достигнуто максимальное количество итераций'
            }
        
        # Извлекаем решение
        solution = np.zeros(n + m)
        for i, var_index in enumerate(basis):
            if var_index < n + m:
                solution[var_index] = tableau[i, -1]
        
        # Значение целевой функции
        max_value = tableau[-1, -1]
        
        return {
            'solution': solution[:n],  # только исходные переменные
            'max_value': max_value,
            'iterations': iteration,
            'success': True,
            'message': 'Оптимальное решение найдено'
        }
        
    except Exception as e:
        return {
            'solution': None,
            'max_value': None,
            'iterations': 0,
            'success': False,
            'message': f'Ошибка: {str(e)}'
        }

def print_tableau(tableau, basis, n_vars):
    """Вспомогательная функция для вывода симплекс-таблицы"""
    print("\nСимплекс-таблица:")
    print("Базис |", end=" ")
    for i in range(tableau.shape[1] - 1):
        if i < n_vars:
            print(f"x{i+1:7}", end=" ")
        elif i < n_vars + len(basis):
            print(f"s{i-n_vars+1:7}", end=" ")
        else:
            print("      ", end=" ")
    print("|  b")
    print("-" * (13 * tableau.shape[1] + 10))
    
    for i in range(tableau.shape[0] - 1):
        if i < len(basis):
            if basis[i] < n_vars:
                print(f"x{basis[i]+1:3} |", end=" ")
            else:
                print(f"s{basis[i]-n_vars+1:3} |", end=" ")
        else:
            print("     |", end=" ")
            
        for j in range(tableau.shape[1]):
            print(f"{tableau[i, j]:8.3f}", end=" ")
        print()
    
    print("-" * (13 * tableau.shape[1] + 10))
    print("z    |", end=" ")
    for j in range(tableau.shape[1]):
        print(f"{tableau[-1, j]:8.3f}", end=" ")
    print()
    print()
