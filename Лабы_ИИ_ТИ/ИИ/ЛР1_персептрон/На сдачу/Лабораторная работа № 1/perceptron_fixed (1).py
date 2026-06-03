import matplotlib
import matplotlib.pyplot as plt
import itertools
import sys

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

class Perceptron:
    def __init__(self, weights):
        self.w = weights

    def net(self, x):
        return sum(xi * wi for xi, wi in zip(x, self.w))

    def Fnet_threshold(self, net):
        return 1 if net >= 0 else 0

    def Fnet_sigmoid(self, net):
        return 0.5 * (net / (1 + abs(net)) + 1)

    def discretize(self, out):
        return 1 if out >= 0.5 else 0

    def update_weights_threshold(self, x, err, lr):
        for i in range(len(self.w)):
            self.w[i] += lr * err * x[i]

    def update_weights_sigmoid(self, x, err_cont, lr, net):
        for i in range(len(self.w)):
            self.w[i] += lr * err_cont * x[i]

def count_errors(perceptron, dataset, activation='threshold'):
    total_error = 0
    for x, t in dataset:
        net = perceptron.net(x)
        if activation == 'threshold':
            out = perceptron.Fnet_threshold(net)
        else:
            out = perceptron.discretize(perceptron.Fnet_sigmoid(net))
        total_error += abs(t - out)
    return total_error

def train_perceptron_threshold(perceptron, dataset, lr, max_epochs):
    epochs_list = []
    errors_list = []

    for epoch in range(max_epochs):
        for x, t in dataset:
            net = perceptron.net(x)
            out = perceptron.Fnet_threshold(net)
            err = t - out
            perceptron.update_weights_threshold(x, err, lr)

        total_error = count_errors(perceptron, dataset, 'threshold')

        epochs_list.append(epoch)
        errors_list.append(total_error)
        print(f"Epoch {epoch}: Weights = {perceptron.w}, Total Error = {total_error}")

        if total_error == 0:
            break

    plt.figure(figsize=(10, 4))
    plt.plot(epochs_list, errors_list, marker='o')
    plt.xlabel('Эпохи')
    plt.ylabel('Ошибки')
    plt.title('Пороговая: Ошибки/эпохи')
    plt.grid()
    plt.savefig('plot_threshold.png', dpi=100)
    plt.show()

def train_perceptron_sigmoid(perceptron, dataset, lr, max_epochs):
    epochs_list = []
    errors_list = []

    for epoch in range(max_epochs):
        for x, t in dataset:
            net = perceptron.net(x)
            out = perceptron.Fnet_sigmoid(net)
            err_cont = t - out
            perceptron.update_weights_sigmoid(x, err_cont, lr, net)

        total_error = count_errors(perceptron, dataset, 'sigmoid')

        epochs_list.append(epoch)
        errors_list.append(total_error)
        print(f"Epoch {epoch}: Weights = {perceptron.w}, Total Error = {total_error}")

        if total_error == 0:
            break

    plt.figure(figsize=(10, 4))
    plt.plot(epochs_list, errors_list, marker='o')
    plt.xlabel('Эпохи')
    plt.ylabel('Ошибки')
    plt.title('Сигмоидная: Ошибки/эпохи')
    plt.grid()
    plt.savefig('plot_sigmoid.png', dpi=100)
    plt.show()

def find_min_vyborka(dataset, activation='threshold', lr=0.3, max_epochs=200):
    n = len(dataset)
    print(f"\nПоиск минимальной выборки [{activation}]:")

    for k in range(1, n + 1):
        found_indices = None
        tried = 0

        for combo in itertools.combinations(range(n), k):
            subset = [dataset[i] for i in combo]
            tried += 1

            p = Perceptron([0.0] * 5)
            if activation == 'threshold':
                train_perceptron_threshold_silent(p, subset, lr, max_epochs)
            else:
                train_perceptron_sigmoid_silent(p, subset, lr, max_epochs)

            if count_errors(p, dataset, activation) == 0:
                found_indices = list(combo)
                break

        total = sum(1 for _ in itertools.combinations(range(n), k))
        status = 'НАЙДЕНА' if found_indices else 'Не найдено'
        print(f"  k={k:2d}: проверено {tried}/{total}  {status}")

        if found_indices:
            print(f"\nМинимальный размер выборки: {k}")
            print(f"Индексы: {found_indices}")
            print("Примеры:")
            for idx in found_indices:
                x, t = dataset[idx]
                xi = [int(v) for v in x[1:]]
                print(f"  x={xi}  t={t}")
            return [dataset[i] for i in found_indices]

    return []

# Тихие версии train-функций (без print и plt) для поиска выборки
def train_perceptron_threshold_silent(perceptron, dataset, lr, max_epochs):
    for epoch in range(max_epochs):
        for x, t in dataset:
            net = perceptron.net(x)
            out = perceptron.Fnet_threshold(net)
            err = t - out
            perceptron.update_weights_threshold(x, err, lr)
        if count_errors(perceptron, dataset, 'threshold') == 0:
            break

def train_perceptron_sigmoid_silent(perceptron, dataset, lr, max_epochs):
    for epoch in range(max_epochs):
        for x, t in dataset:
            net = perceptron.net(x)
            out = perceptron.Fnet_sigmoid(net)
            err_cont = t - out
            perceptron.update_weights_sigmoid(x, err_cont, lr, net)
        if count_errors(perceptron, dataset, 'sigmoid') == 0:
            break

if __name__ == "__main__":
    log_file = open('results.txt', 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_file)

    dataset = generate_dataset()

    print("--- Пороговая ---")
    p_thresh = Perceptron([0.0] * 5)
    train_perceptron_threshold(p_thresh, dataset, lr=0.3, max_epochs=100)

    print("\n--- Сигмоид ---")
    p_sigmoid = Perceptron([0.0] * 5)
    train_perceptron_sigmoid(p_sigmoid, dataset, lr=0.3, max_epochs=100)

    print("\n--- Минимальная выборка для пороговой ---")
    min_vyborka_thresh = find_min_vyborka(dataset, activation='threshold')

    print("\n--- Минимальная выборка для сигмоидной ---")
    min_vyborka_sigmoid = find_min_vyborka(dataset, activation='sigmoid')

    log_file.close()
    sys.stdout = sys.__stdout__
    print("\nРезультаты сохранены в results.txt")
