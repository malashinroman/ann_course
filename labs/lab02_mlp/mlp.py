import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List

# --- Константы для выбора функции активации ---
SIGMOID = 'sigmoid'
RELU = 'relu'

# --- Реализации функций активации и их производных ---
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Сигмоида"""
    raise NotImplementedError("Необходимо реализовать сигмоиду!")

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Производная сигмоида"""
    raise NotImplementedError("Необходимо реализовать производную сигмоиды!")

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU (Rectified Linear Unit)"""
    raise NotImplementedError("Необходимо реализовать ReLU!")

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Производная ReLU"""
    raise NotImplementedError("Необходимо реализовать производную ReLU!")


# --- Сопоставление констант с функциями активации ---
activation_map: dict = {
    SIGMOID: (sigmoid, sigmoid_derivative),
    RELU: (relu, relu_derivative)
}

class MLP:
    """
    Простой многослойный персептрон (MLP)

    Args:
        layer_sizes (List(int)):
            Список из числа входных, скрытых и выходных нейронов
        activation_name (str):
            Константа для выбора функции активации (SIGMOID, RELU)
        learning_rate (float):
            Скорость обучения (коэффициент градиентного спуска)
    """
    def __init__(
        self,
        layer_sizes: List[int],
        activation_name: str,
        learning_rate: float = 0.1
    ) -> None:
        assert activation_name in activation_map, f"Неподдерживаемая активация: {activation_name}"
        
        # Параметры сети
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        
        # Выбираем функции активации
        self.activation_fn, self.activation_derivative_fn = activation_map[activation_name]
        
        # Инициализация весов и смещений
        self.weights: List[np.ndarray] = []  # W[l]: матрица весов для перехода l->l+1
        self.biases: List[np.ndarray] = []   # b[l]: вектор смещений для слоя l+1
        for in_neurons, out_neurons in zip(layer_sizes[:-1], layer_sizes[1:]):
            weight_matrix = np.random.randn(out_neurons, in_neurons) * 0.1
            bias_vector = np.zeros((out_neurons, 1))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)


    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Прямой проход по сети

        Args:
            inputs (np.ndarray): Входные данные формой (n_features, n_samples)
            
        Returns:
            (np.ndarray): Предсказанные значения (вероятности после сигмоиды)
        """
        self.activations: List[np.ndarray] = [inputs]
        self.z: List[np.ndarray] = []
        
        last_layer_idx = len(self.layer_sizes) - 1
        
        # Для каждого слоя считаем Z = W·A + b, затем A = activation_fn(Z)
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = W @ self.activations[idx] + b
            self.z.append(Z)
            
            A = self.activation_fn(Z) if idx != last_layer_idx else Z
            self.activations.append(A)
        return sigmoid(self.activations[-1])


    def backward(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Обратное распространение ошибки и обновление весов

        Args:
            inputs (np.ndarray): Входные данные формой (n_features, n_samples)
            targets (np.ndarray): Эталонные значения формой (1, n_samples)
        """
        raise NotImplementedError("Необходимо реализовать алгоритм обратного распространения ошибки!")


    def train(self, inputs: np.ndarray, targets: np.ndarray,
              epochs: int = 1000, print_interval: int = 100) -> None:
        """
        Обучение сети

        Args:
            inputs (np.ndarray): Входные данные формой (n_features, n_samples)
            targets (np.ndarray): Эталонные значения формой (1, n_samples)
            epochs (int): Число эпох обучения
            print_interval (int): Как часто печатать прогресс
        """
        for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
            # Прямой и обратный проход
            predictions = self.forward(inputs)
            loss = np.mean((targets - predictions) ** 2)
            
            self.backward(inputs, targets)
            
            # Печать лосса и точности
            if epoch == 1 or epoch % print_interval == 0:
                print(predictions.max(), predictions.min(), predictions.mean())
                predicted_labels = (predictions > 0.5).astype(int)
                accuracy = np.mean(predicted_labels == targets)
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}%")


    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Возвращает предсказанные значения (вероятности)
        
        Args:
            inputs (np.ndarray): Входные данные формой (n_features, n_samples)
        """
        return self.forward(inputs)


if __name__ == "__main__":
    # Размер каждого сегмента данных
    sample_count = 2000
    
    # Генерируем 4 группы точек
    bottom_left = np.random.uniform(0, 0.5, (sample_count, 2))
    top_right = np.random.uniform(0.5, 1.0, (sample_count, 2))
    top_left = np.random.uniform(0, 0.5, (sample_count, 2))
    top_left[:,1] += 0.5
    bottom_right = np.random.uniform(0.5, 1.0, (sample_count, 2))
    bottom_right[:,1] -= 0.5
    
    # Объединяем в классы
    class_zero = np.vstack((bottom_left, top_right))
    class_one = np.vstack((top_left, bottom_right))
    
    # Собираем всю выборку и метки
    data = np.vstack((class_zero, class_one))  # shape: (4*sample_count, 2)
    labels = np.concatenate((np.zeros((2*sample_count, 1)), np.ones((2*sample_count, 1))), axis=0)
    
    # Перемешиваем данные и метки одним порядком и разбиваем на train/test (3:1)
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_point = int(num_samples * 0.75)
    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]
    train_data, train_labels = data[train_idx], labels[train_idx]
    test_data,  test_labels  = data[test_idx],  labels[test_idx]

    # Транспонируем для модели: (features, samples)
    X_train, Y_train = train_data.T, train_labels.T
    X_test,  Y_test  = test_data.T,  test_labels.T

    output_folder = 'outputs'
    os.makedirs(output_folder, exist_ok=True)

    # Визуализация обучающих данных
    plt.scatter(train_data[:,0], train_data[:,1], c='gray', alpha=0.5, label='Train')
    plt.scatter(test_data[:,0],  test_data[:,1],  c='black', alpha=0.5, label='Test')
    plt.legend(loc='upper right')
    plt.title("Train/Test разбивка (75%/25%)")
    plt.show()
    plt.scatter(class_zero[:,0], class_zero[:,1], c='red', label='Класс 0')
    plt.scatter(class_one[:,0], class_one[:,1], c='blue', label='Класс 1')
    plt.legend(loc='upper right')
    plt.title("Исходные данные")
    plt.savefig(f"{output_folder}/source_data.png")
    plt.show()
    
    # Создаем и обучаем модель
    # layer_sizes: [2, 10, 10, 1] ->  2 входных признака, 10 нейронов в первом скрытом слое,
    #                                 10 нейронов во втором скрытом слое, 1 выходной нейрон
    #                                 (можно добавлять сколько угодно скрытых слоев произвольной размерности)
    # activation_name: RELU        # можно заменить на SIGMOID
    # learning_rate: 0.1           # скорость обучения, чем меньше, тем более мелкие шаги при АОРО на обучении
    model = MLP(
        layer_sizes=[2, 10, 10, 1],
        activation_name=RELU,
        learning_rate=0.1
    )
    model.train(X_train, Y_train, epochs=10000, print_interval=1000)
    
    # Предсказания и визуализация результатов
    predictions = model.predict(X_test).reshape(-1)
    pred_colors = ['blue' if p > 0.5 else 'red' for p in predictions]
    plt.scatter(test_data[:,0], test_data[:,1], c=pred_colors)
    plt.title("Предсказанные данные")
    plt.savefig(f"{output_folder}/predicted_data.png")
    plt.show()
    
    # Визуализация ошибок: зелёный = ошибочно классифицированные точки
    error_colors = ['black' if ((predictions[j] > 0.5) != labels[i,0]) else
                    ('blue' if predictions[j] > 0.5 else 'red')
                    for i, j in zip(test_idx, range(len(test_idx)))]
    plt.scatter(test_data[:,0], test_data[:,1], c=error_colors)
    plt.title("Ошибки при классификации (чёрный)")
    plt.savefig(f"{output_folder}/errors.png")
    plt.show()