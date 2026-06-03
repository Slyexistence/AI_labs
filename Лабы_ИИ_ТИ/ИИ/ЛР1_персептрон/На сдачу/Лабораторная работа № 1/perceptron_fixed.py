"""
Перцептрон — исправленная версия с поиском минимальной выборки.

Исправления относительно оригинала:
  1. sigmoid: ошибка эпохи считается ПОСЛЕ обновления всех весов отдельным
     проходом, а не накопленно во время обновлений (иначе total_error == 0
     не гарантирует правильную классификацию).
  2. plt.show() заменён на plt.savefig() — графики сохраняются в файлы.
  3. Весь вывод дублируется в текстовый файл через класс Tee.
  4. Добавлен find_minimum_sample() — поиск минимальной обучающей выборки
     перебором через itertools.combinations.
"""

import matplotlib
matplotlib.use('Agg')               # без GUI-окна
import matplotlib.pyplot as plt
import itertools
import sys
import os
from datetime import datetime

OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# Tee: одновременный вывод в stdout и файл
# ────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ────────────────────────────────────────────────────────────────────
# Булева функция и генерация датасета
# ────────────────────────────────────────────────────────────────────
def boolean_function(x1, x2, x3, x4):
    part1 = x1 or x2 or x3
    part2 = x2 or x3 or x4
    return int(not (part1 and part2))


def generate_dataset():
    dataset = []
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                for x4 in (0, 1):
                    x = [1.0, float(x1), float(x2), float(x3), float(x4)]
                    t = boolean_function(x1, x2, x3, x4)
                    dataset.append((x, t))
    return dataset


# ────────────────────────────────────────────────────────────────────
# Перцептрон
# ────────────────────────────────────────────────────────────────────
class Perceptron:
    def __init__(self, weights):
        self.w = list(weights)

    def net(self, x):
        return sum(xi * wi for xi, wi in zip(x, self.w))

    def Fnet_threshold(self, net_val):
        return 1 if net_val >= 0 else 0

    def Fnet_sigmoid(self, net_val):
        return 0.5 * (net_val / (1 + abs(net_val)) + 1)

    def discretize(self, out):
        return 1 if out >= 0.5 else 0

    def update_weights_threshold(self, x, err, lr):
        for i in range(len(self.w)):
            self.w[i] += lr * err * x[i]

    def update_weights_sigmoid(self, x, err_cont, lr):
        for i in range(len(self.w)):
            self.w[i] += lr * err_cont * x[i]


# ────────────────────────────────────────────────────────────────────
# Вспомогательная: подсчёт дискретных ошибок на датасете
# ────────────────────────────────────────────────────────────────────
def count_errors(perceptron, dataset, activation='threshold'):
    errors = 0
    for x, t in dataset:
        net_val = perceptron.net(x)
        if activation == 'threshold':
            out = perceptron.Fnet_threshold(net_val)
        else:
            out = perceptron.discretize(perceptron.Fnet_sigmoid(net_val))
        if out != t:
            errors += 1
    return errors


# ────────────────────────────────────────────────────────────────────
# Обучение — пороговая функция
# ────────────────────────────────────────────────────────────────────
def train_threshold(perceptron, dataset, lr=0.3, max_epochs=100,
                    verbose=True, plot_title='Пороговая', save_path=None):
    epochs_list, errors_list = [], []

    for epoch in range(max_epochs):
        # Обновление весов (онлайн)
        for x, t in dataset:
            net_val = perceptron.net(x)
            out = perceptron.Fnet_threshold(net_val)
            err = t - out
            perceptron.update_weights_threshold(x, err, lr)

        # ── ИСПРАВЛЕНИЕ ─────────────────────────────────────────────
        # Ошибки считаются отдельным проходом с текущими весами,
        # а не накопленно во время обновлений.
        total_error = count_errors(perceptron, dataset, 'threshold')
        # ────────────────────────────────────────────────────────────

        epochs_list.append(epoch)
        errors_list.append(total_error)

        if verbose:
            w_str = [f'{wi:.4f}' for wi in perceptron.w]
            print(f"  Epoch {epoch:3d}: w={w_str}  Total Error={total_error}")

        if total_error == 0:
            break

    if save_path:
        plt.figure(figsize=(10, 4))
        plt.plot(epochs_list, errors_list, marker='o', color='steelblue')
        plt.xlabel('Эпохи')
        plt.ylabel('Ошибки')
        plt.title(plot_title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  → График: {save_path}")

    return epochs_list, errors_list


# ────────────────────────────────────────────────────────────────────
# Обучение — сигмоидная функция
# ────────────────────────────────────────────────────────────────────
def train_sigmoid(perceptron, dataset, lr=0.3, max_epochs=100,
                  verbose=True, plot_title='Сигмоидная', save_path=None):
    epochs_list, errors_list = [], []

    for epoch in range(max_epochs):
        # Обновление весов (онлайн, непрерывная ошибка)
        for x, t in dataset:
            net_val = perceptron.net(x)
            out = perceptron.Fnet_sigmoid(net_val)
            err_cont = t - out
            perceptron.update_weights_sigmoid(x, err_cont, lr)

        # ── ИСПРАВЛЕНИЕ ─────────────────────────────────────────────
        # Дискретные ошибки считаются с ТЕКУЩИМИ весами (после всех
        # обновлений эпохи), а не накопленно во время обновлений.
        # Это гарантирует: total_error == 0 ↔ всё правильно.
        total_error = count_errors(perceptron, dataset, 'sigmoid')
        # ────────────────────────────────────────────────────────────

        epochs_list.append(epoch)
        errors_list.append(total_error)

        if verbose:
            w_str = [f'{wi:.4f}' for wi in perceptron.w]
            print(f"  Epoch {epoch:3d}: w={w_str}  Total Error={total_error}")

        if total_error == 0:
            break

    if save_path:
        plt.figure(figsize=(10, 4))
        plt.plot(epochs_list, errors_list, marker='o', color='darkorange')
        plt.xlabel('Эпохи')
        plt.ylabel('Ошибки')
        plt.title(plot_title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  → График: {save_path}")

    return epochs_list, errors_list


# ────────────────────────────────────────────────────────────────────
# Поиск минимальной обучающей выборки
# ────────────────────────────────────────────────────────────────────
def find_minimum_sample(full_dataset, activation='threshold',
                         lr=0.3, max_epochs=200):
    """
    Перебирает подмножества full_dataset размером k = 1, 2, …
    Обучает перцептрон на подмножестве, проверяет на всех 16 примерах.
    Возвращает (min_k, индексы_примеров, финальные_веса).
    """
    n = len(full_dataset)
    print(f"\n  Поиск минимальной выборки [{activation}]:")

    for k in range(1, n + 1):
        found_combo = None
        found_weights = None
        tried = 0

        for combo in itertools.combinations(range(n), k):
            subset = [full_dataset[i] for i in combo]
            tried += 1

            p = Perceptron([0.0] * 5)
            if activation == 'threshold':
                train_threshold(p, subset, lr, max_epochs, verbose=False)
            else:
                train_sigmoid(p, subset, lr, max_epochs, verbose=False)

            if count_errors(p, full_dataset, activation) == 0:
                found_combo = combo
                found_weights = list(p.w)
                break

        total = sum(1 for _ in itertools.combinations(range(n), k))
        status = '✓ НАЙДЕНА' if found_combo else '✗'
        print(f"  k={k:2d}: проверено {tried}/{total}  {status}")

        if found_combo:
            indices = list(found_combo)
            print(f"\n  Минимальный размер выборки: {k}")
            print(f"  Индексы примеров:  {indices}")
            print(f"  Финальные веса:    {[f'{w:.4f}' for w in found_weights]}")
            print(f"  Примеры:")
            for idx in indices:
                x, t = full_dataset[idx]
                xi = [int(v) for v in x[1:]]
                print(f"    {xi}  →  t={t}")
            return k, indices, found_weights

    return None, None, None


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────
def main():
    txt_path = os.path.join(OUTPUT_DIR, 'perceptron_results.txt')

    with open(txt_path, 'w', encoding='utf-8') as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)

        print("=" * 68)
        print("  ПЕРЦЕПТРОН | ОБУЧЕНИЕ + МИНИМАЛЬНАЯ ВЫБОРКА")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 68)

        full_dataset = generate_dataset()

        # ── 1. Таблица истинности ───────────────────────────────────
        print("\n[1] Таблица истинности  f = NOT((x1|x2|x3) AND (x2|x3|x4))")
        print(f"    {'x1':>3}{'x2':>4}{'x3':>4}{'x4':>4}{'f':>4}")
        print("    " + "─" * 18)
        for x, t in full_dataset:
            xi = [int(v) for v in x[1:]]
            print(f"    {'  '.join(map(str, xi))}  {t}")

        # ── 2. Обучение на полном датасете ──────────────────────────
        print("\n" + "=" * 68)
        print("[2] ОБУЧЕНИЕ НА ПОЛНОМ ДАТАСЕТЕ (16 примеров)")

        print("\n--- Пороговая ---")
        p_t = Perceptron([0.0] * 5)
        ep_t, er_t = train_threshold(
            p_t, full_dataset, lr=0.3, max_epochs=100,
            plot_title='Пороговая (полный датасет)',
            save_path=os.path.join(OUTPUT_DIR, 'plot_threshold_full.png'))
        print(f"  Итог: ошибок={count_errors(p_t, full_dataset, 'threshold')} | "
              f"эпох={len(ep_t)} | "
              f"Сходимость={'ДА' if er_t[-1] == 0 else 'НЕТ'}")

        print("\n--- Сигмоидная ---")
        p_s = Perceptron([0.0] * 5)
        ep_s, er_s = train_sigmoid(
            p_s, full_dataset, lr=0.3, max_epochs=100,
            plot_title='Сигмоидная (полный датасет)',
            save_path=os.path.join(OUTPUT_DIR, 'plot_sigmoid_full.png'))
        print(f"  Итог: ошибок={count_errors(p_s, full_dataset, 'sigmoid')} | "
              f"эпох={len(ep_s)} | "
              f"Сходимость={'ДА' if er_s[-1] == 0 else 'НЕТ'}")

        # ── 3. Поиск минимальной выборки ────────────────────────────
        print("\n" + "=" * 68)
        print("[3] ПОИСК МИНИМАЛЬНОЙ ОБУЧАЮЩЕЙ ВЫБОРКИ")

        k_t, idx_t, w_t = find_minimum_sample(full_dataset, 'threshold')
        k_s, idx_s, w_s = find_minimum_sample(full_dataset, 'sigmoid')

        # ── 4. Обучение на минимальной выборке ──────────────────────
        print("\n" + "=" * 68)
        print("[4] ОБУЧЕНИЕ НА МИНИМАЛЬНОЙ ВЫБОРКЕ + ПРОВЕРКА")

        if idx_t:
            subset_t = [full_dataset[i] for i in idx_t]
            print(f"\n--- Пороговая | k={k_t} ---")
            pm_t = Perceptron([0.0] * 5)
            train_threshold(pm_t, subset_t, lr=0.3, max_epochs=200,
                plot_title=f'Пороговая (мин. выборка k={k_t})',
                save_path=os.path.join(OUTPUT_DIR, f'plot_threshold_min{k_t}.png'))
            print(f"  Ошибок на полном датасете: "
                  f"{count_errors(pm_t, full_dataset, 'threshold')}")

        if idx_s:
            subset_s = [full_dataset[i] for i in idx_s]
            print(f"\n--- Сигмоидная | k={k_s} ---")
            pm_s = Perceptron([0.0] * 5)
            train_sigmoid(pm_s, subset_s, lr=0.3, max_epochs=200,
                plot_title=f'Сигмоидная (мин. выборка k={k_s})',
                save_path=os.path.join(OUTPUT_DIR, f'plot_sigmoid_min{k_s}.png'))
            print(f"  Ошибок на полном датасете: "
                  f"{count_errors(pm_s, full_dataset, 'sigmoid')}")

        # ── 5. Сводка ────────────────────────────────────────────────
        print("\n" + "=" * 68)
        print("[5] ИТОГОВАЯ СВОДКА")
        print(f"  Датасет: 16 примеров, 3 класса 1 / 13 класса 0")
        print(f"  {'Метод':<15} {'Эпох (полный)':<18} {'Мин. выборка k':<18} Сходимость")
        print(f"  {'─'*60}")
        print(f"  {'Пороговая':<15} {len(ep_t):<18} {k_t if k_t else '?':<18} "
              f"{'ДА' if er_t[-1]==0 else 'НЕТ'}")
        print(f"  {'Сигмоидная':<15} {len(ep_s):<18} {k_s if k_s else '?':<18} "
              f"{'ДА' if er_s[-1]==0 else 'НЕТ'}")
        print("=" * 68)
        print(f"\nРезультаты → {txt_path}")
        print(f"Графики    → {OUTPUT_DIR}/")

        sys.stdout = sys.__stdout__

    print(f"\n[DONE] {txt_path}")


if __name__ == "__main__":
    main()
