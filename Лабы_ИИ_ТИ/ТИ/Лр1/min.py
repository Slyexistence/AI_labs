import numpy as np
import pandas as pd
from fractions import Fraction

class Simplex:
    def __init__(self, c, A, b, M=1000):
        """
        Инициализация симплекс таблицы для М-метода
        
        Parameters:
        c: коэффициенты целевой функции (минимизация)
        A: матрица ограничений
        b: вектор правых частей (>= 0)
        M: большое число для штрафа
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.M = M
        self.num_original_vars = len(c)
        self.num_constraints = len(A)
        
        # Проверка на неотрицательность b
        for i in range(self.num_constraints):
            if self.b[i] < 0:
                self.A[i] = -self.A[i]
                self.b[i] = -self.b[i]
        
        self.build_initial_table()
    
    def build_initial_table(self):
        """Построение начальной симплекс таблицы с искусственными переменными"""
        
        # Общее количество переменных:
        # оригинальные + избыточные (для >=) + искусственные
        self.num_slack = self.num_constraints  # для ограничений >= нужны избыточные (-1)
        self.num_artificial = self.num_constraints  # искусственные переменные
        
        self.total_vars = self.num_original_vars + self.num_slack + self.num_artificial
        
        # Индексы переменных:
        # 0..n-1: оригинальные
        # n..n+m-1: избыточные (с коэффициентом -1)
        # n+m..n+2m-1: искусственные
        
        # Создаем расширенную матрицу ограничений
        self.tableau = np.zeros((self.num_constraints + 2, self.total_vars + 1))
        
        # Заполняем коэффициенты ограничений
        for i in range(self.num_constraints):
            # Оригинальные переменные
            self.tableau[i, :self.num_original_vars] = self.A[i]
            
            # Избыточные переменные (-1 для ограничений >=)
            self.tableau[i, self.num_original_vars + i] = -1
            
            # Искусственные переменные (+1)
            self.tableau[i, self.num_original_vars + self.num_slack + i] = 1
            
            # Правая часть
            self.tableau[i, -1] = self.b[i]
        
        # Заполняем целевую функцию (Z - строка)
        # Для минимизации с искусственными переменными используем M
        # Z = sum(c_j * x_j) + M * sum(artificial)
        
        # Коэффициенты оригинальных переменных
        self.tableau[-2, :self.num_original_vars] = self.c
        
        # Коэффициенты искусственных переменных (M)
        for i in range(self.num_artificial):
            self.tableau[-2, self.num_original_vars + self.num_slack + i] = self.M
        
        # Вычисляем Z для начального базиса (искусственные переменные в базисе)
        for i in range(self.num_constraints):
            if self.tableau[i, self.num_original_vars + self.num_slack + i] == 1:
                # Для каждой искусственной переменной в базисе вычитаем M * строку
                self.tableau[-2] -= self.M * self.tableau[i]
        
        # W-строка для проверки оптимальности
        self.tableau[-1] = self.tableau[-2].copy()
        
        # Базисные переменные
        self.basis = []
        for i in range(self.num_constraints):
            # В начальном базисе только искусственные переменные
            self.basis.append(self.num_original_vars + self.num_slack + i)
    
    def format_number(self, x): # Форматирование для читаемого вида
        if abs(x) < 1e-10:
            return "0"
        elif abs(x - round(x)) < 1e-10:
            return str(int(x))
        else:
            # Используем дроби для точности
            frac = Fraction(x).limit_denominator(100)
            return str(frac)
    
    def display_tableau(self, iteration):
        """Вывод текущей симплекс таблицы в отформатированном виде"""
        print(f"\n{'.'*80}")
        print(f"ИТЕРАЦИЯ {iteration}")
        print(f"{'.'*80}")
        
        # Создаем заголовки для переменных
        headers = []
        for i in range(self.num_original_vars):
            headers.append(f'x{i+1}')
        for i in range(self.num_slack):
            headers.append(f's{i+1}')
        for i in range(self.num_artificial):
            headers.append(f'r{i+1}')
        headers.append('RHS')
        
        # Создаем индексы строк
        row_names = []
        for i in range(self.num_constraints):
            var_idx = self.basis[i]
            if var_idx < self.num_original_vars:
                row_names.append(f'x{var_idx+1}')
            elif var_idx < self.num_original_vars + self.num_slack:
                row_names.append(f's{var_idx - self.num_original_vars + 1}')
            else:
                row_names.append(f'r{var_idx - self.num_original_vars - self.num_slack + 1}')
        row_names.append('Z')
        row_names.append('W')
        
        # Создаем DataFrame для красивого вывода
        df = pd.DataFrame(self.tableau.round(3), 
                         index=row_names, 
                         columns=headers)
        
        print(df.to_string())
        print(f"Текущее значение Z: {self.format_number(-self.tableau[-2, -1])}")
        print(f"Текущее значение W: {self.format_number(-self.tableau[-1, -1])}")
    
    def pivot(self):
        """Выполнение одной итерации симплекс-метода"""
        
        # Проверяем W-строку на оптимальность
        w_row = self.tableau[-1, :-1]
        
        # Находим наименьший коэффициент в W-строке (для минимизации)
        # Если все коэффициенты >= 0 с учетом точности, то оптимальное решение найдено
        if np.all(w_row >= -1e-10):
            return False
        
        # Выбираем ведущий столбец (наименьший коэффициент в W-строке)
        pivot_col = np.argmin(w_row)
        
        print(f"\nВыбран ведущий столбец: {pivot_col + 1} (коэффициент = {self.format_number(w_row[pivot_col])})")
        
        # Находим ведущую строку по минимальному отношению
        ratios = []
        for i in range(self.num_constraints):
            if self.tableau[i, pivot_col] > 1e-10:
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                ratios.append((ratio, i))
        
        if not ratios:
            raise Exception("Задача не имеет допустимого решения (неограниченная)")
        
        # Выбираем строку с минимальным отношением
        pivot_row = min(ratios)[1]
        pivot_element = self.tableau[pivot_row, pivot_col]
        
        print(f"Выбрана ведущая строка: {pivot_row + 1} (ведущий элемент = {self.format_number(pivot_element)})")
        
        # Обновляем базис
        self.basis[pivot_row] = pivot_col
        
        # Нормализуем ведущую строку
        self.tableau[pivot_row] = self.tableau[pivot_row] / pivot_element
        
        # Обновляем остальные строки (включая Z и W)
        for i in range(self.num_constraints + 2):
            if i != pivot_row:
                factor = self.tableau[i, pivot_col]
                if abs(factor) > 1e-10:
                    self.tableau[i] -= factor * self.tableau[pivot_row]
        
        return True
    
    def solve(self, max_iterations=100):
        """Решение задачи симплекс М-методом"""
        
        print("\n" + "."*80)
        print("НАЧАЛЬНАЯ СИМПЛЕКС ТАБЛИЦА")
        print("."*80)
        print("\nИскусственные переменные: r1, r2, ..., rm")
        print("Избыточные переменные: s1, s2, ..., sm")
        print("M =", self.M)
        
        self.display_tableau(0)
        
        iteration = 1
        while iteration <= max_iterations:
            if not self.pivot():
                break
            self.display_tableau(iteration)
            iteration += 1
        
        if iteration > max_iterations:
            print("\nДостигнуто максимальное число итераций")
        
        self.print_solution()
        
        return self.get_solution()
    
    def print_solution(self):
        """Вывод оптимального решения"""
        print("\n" + "."*80)
        print("ОПТИМАЛЬНОЕ РЕШЕНИЕ")
        print("."*80)
        
        # Проверяем, есть ли искусственные переменные в базисе
        artificial_in_basis = False
        for var_idx in self.basis:
            if var_idx >= self.num_original_vars + self.num_slack:
                artificial_in_basis = True
                break
        
        if artificial_in_basis:
            # Проверяем значение W
            if abs(self.tableau[-1, -1]) > 1e-6:
                print("Задача не имеет допустимого решения!")
                return
            else:
                print("Предупреждение: Искусственные переменные в базисе с нулевым значением")
        
        # Получаем значения переменных
        solution = np.zeros(self.total_vars)
        for i, var_idx in enumerate(self.basis):
            solution[var_idx] = self.tableau[i, -1]
        
        # Выводим оригинальные переменные
        print("\nЗначения переменных:")
        for i in range(self.num_original_vars):
            print(f"x{i+1} = {self.format_number(solution[i])}")
        
        # Выводим избыточные переменные
        for i in range(self.num_slack):
            print(f"s{i+1} = {self.format_number(solution[self.num_original_vars + i])}")
        
        # Выводим значение целевой функции
        z_value = -self.tableau[-2, -1]
        print(f"\nОптимальное значение Z (минимум) = {self.format_number(z_value)}")
    
    def get_solution(self):
        """Возвращает решение в виде списка"""
        solution = np.zeros(self.num_original_vars + 1)
        
        # # Оригинальные переменные
        # for i in range(self.num_original_vars):
        #     solution[f'x{i+1}'] = 0
        
        # Получаем значения из базиса
        for i, var_idx in enumerate(self.basis):
            if var_idx < self.num_original_vars:
                # solution[f'x{var_idx+1}'] = self.tableau[i, -1]
                solution[var_idx] = self.tableau[i, -1]
        
        solution[-1] = -self.tableau[-2, -1]
        
        return solution


def simplex_min(c, A, b, M=1000, display=True):
    """
    Решение задачи линейного программирования с минимизацией с помощью симплекс М-метода
    """
    
    print("\n" + "."*80)
    print("РЕШЕНИЕ ЗАДАЧИ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
    print("МЕТОД: СИМПЛЕКС М-МЕТОД")
    print("ТИП: МИНИМИЗАЦИЯ")
    print("ОГРАНИЧЕНИЯ: ВСЕ ≥")
    print("."*80)
    
    print("\nИсходная задача:")
    print("Минимизировать Z =", " + ".join([f"{c[i]}*x{i+1}" for i in range(len(c))]))
    print("При ограничениях:")
    for i in range(len(A)):
        constraint = " + ".join([f"{A[i][j]}*x{j+1}" for j in range(len(c))])
        print(f"{constraint} ≥ {b[i]}")
    print(f"xⱼ ≥ 0 для всех j")
    
    # Создаем и решаем задачу
    solver = Simplex(c, A, b, M)
    solution = solver.solve()
    
    return solution
