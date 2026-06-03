import numpy as np
from Код.game import game_simplex

if __name__ == "__main__":
    # Вариант 16
    C_matrix = [
        [16, 17,  8, 15, 17],
        [ 0,  3, 19,  8,  2],
        [13, 19,  7, 15,  9],
        [11, 15,  2, 16,  2]
    ]
    
    result = game_simplex(C_matrix)

    print("\n" + "."*15 + " Конечный результат " + "."*30)
    print(f"Статус: {result['status']}")
    print(f"Метод: {result['method_used']}")
    print(f"Цена игры V: {result['game_value']:.3f}")
    print(f"Оптимальная стратегия A : {np.round(result['strategy_A'], 3)}")
    print(f"Оптимальная стратегия B : {np.round(result['strategy_B'], 3)}")
  
