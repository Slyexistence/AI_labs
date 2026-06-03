import numpy as np
from scipy.optimize import linprog
from Код.min import simplex_min
from Код.max import simplex_max

def game_simplex(payoff_matrix):
    C_matrix = np.array(payoff_matrix, dtype=float)
    rows, columns = C_matrix.shape

    print("."*50)
    print("Исходная платежная матрица (игрок A - строки, B - столбцы):")
    print(C_matrix)
    print("."*50)

    # Проверка на наличие седловой точки в чистых стратегиях
    # Нижняя цена игры
    max_of_row_mins = np.max(np.min(C_matrix, axis=1))
    # Верхняя цена игры
    min_of_col_maxs = np.min(np.max(C_matrix, axis=0))

    print(f"Нижняя цена игры (maximin): {max_of_row_mins}")
    print(f"Верхняя цена игры (minimax): {min_of_col_maxs}")

    if np.isclose(max_of_row_mins, min_of_col_maxs):
        game_value = max_of_row_mins
        print(f"✓ Найдена седловая точка! Цена игры V = {game_value}")
        # Находим индексы седловой точки (для примера)
        row_idx = np.where(np.min(C_matrix) == game_value)[0][0]
        col_idx = np.where(np.max(C_matrix) == game_value)[0][0]
        # Чистые стратегии - это единичные векторы
        strategy_A = np.zeros(rows)
        strategy_A[row_idx] = 1.0
        strategy_B = np.zeros(columns)
        strategy_B[col_idx] = 1.0

        return {
            'game_value': game_value,
            'strategy_A': strategy_A,
            'strategy_B': strategy_B,
            'method_used': 'pure (saddle point)',
            'status': 'success'
        }
    else:
        print("✗ Седловой точки нет. Переходим к поиску смешанных стратегий.")
        print("."*50)

    # Подготовка к решению задачи ЛП (получение положительных значений в матрице)
    min_val = np.min(C_matrix)
    T = 0
    if min_val <= 0:
        T = abs(min_val) + 1  # Прибавляем 1, чтобы гарантированно сделать все элементы > 0
        C_transformed = C_matrix + T
        print(f"Матрица содержит неположительные элементы. Выполнено преобразование: C' = C + {T}")
        print("Преобразованная матрица C':")
        print(C_transformed)
    else:
        C_transformed = C_matrix
        print("Матрица уже неотрицательная.")
    print("."*50)


    # Решение задачи ЛП для игрока A (минимизация)
    # Целевая функция: W = u1 + u2 + ... + um -> min
    # Ограничения: sum( c_ij * u_i ) >= 1  для всех j
    #              u_i >= 0
    # Переменные: u = [u1, u2, ..., um]

    # В linprog для минимизации: c @ u -> min
    c_A = np.ones(rows)  # Коэффициенты для суммы u_i

    # Матрица ограничений A_ub @ u <= b_ub. Нам нужно C.T @ u >= 1.
    # Умножаем на -1: -C.T @ u <= -1 (так как linprog не принимает неравенства >=)
    A_ub_A = -C_transformed.T
    b_ub_A = -np.ones(columns)

    print("Решаем задачу ЛП для игрока A (минимизация суммы u_i)...")
    result_A = simplex_min(c_A, A_ub_A, b_ub_A)

    u_opt = result_A[0:-1]
    W_opt = result_A[-1]

    print(f"Оптимальное решение (u*): {u_opt.round(3)}")
    print(f"Минимальное значение W = {W_opt:.3f}")

    # Расчет стратегии и цены игры для A
    # Цена преобразованной игры V' = 1 / W_opt
    # Стратегия A: x_i* = u_i* / W_opt
    game_value_transformed = 1.0 / W_opt
    strategy_A = u_opt / W_opt

    print(f"Цена преобразованной игры V' = {game_value_transformed:.3f}")
    print(f"Оптимальная стратегия игрока A (x*): {strategy_A.round(3)}")
    print("."*50)


    # Решение задачи ЛП для игрока B (максимизация)
    # Целевая функция: Z = v1 + v2 + ... + vn -> max
    # Ограничения: sum( c_ij * v_j ) <= 1  для всех i
    #              v_j >= 0

    c_B = np.ones(columns)

    # Матрица ограничений: C_transformed @ v <= 1
    A_ub_B = C_transformed
    b_ub_B = np.ones(rows)

    print("Решаем задачу ЛП для игрока B (максимизация суммы v_j)...")
    result_B = simplex_max(c_B, A_ub_B, b_ub_B)

    v_opt = result_B["solution"]
    # Z_opt = -result_B.fun, так как мы минимизировали -Z
    Z_opt = result_B["max_value"]

    print(f"Оптимальное решение (v*): {v_opt.round(3)}")
    print(f"Максимальное значение Z = {Z_opt:.3f}")

    # Расчет стратегии для B и проверка
    # Стратегия B: y_j* = v_j* / Z_opt
    # Цена преобразованной игры также V' = 1 / Z_opt (должна совпасть)
    strategy_B = v_opt / Z_opt

    print(f"Цена преобразованной игры V' (из задачи B) = {1/Z_opt:.3f}")
    print(f"Оптимальная стратегия игрока B (y*): {strategy_B.round(3)}")
    print("."*50)

    # Возврат к исходной цене игры
    original_game_value = game_value_transformed - T
    print(f"Возврат к исходной цене игры: V = V' - {T} = {original_game_value:.3f}")

    return {
        'game_value': original_game_value,
        'strategy_A': strategy_A,
        'strategy_B': strategy_B,
        'method_used': 'mixed (LP solution)',
        'status': 'success'
    }
