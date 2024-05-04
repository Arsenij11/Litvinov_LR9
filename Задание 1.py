import torch.nn as nn
import torch.optim as optim

# Определение архитектуры нейронной сети
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Начальные параметры
input_size = 10  # Пример размерности входных данных
hidden_size = 40  # Число нейронов в скрытом слое
output_size = 1  # Пример размерности выходных данных
lr = 0.01  # Шаг градиентного спуска

# Создание нейронной сети
model = NeuralNetwork(input_size, hidden_size, output_size)

# Определение функции потерь и метода оптимизации
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Вывод архитектуры модели
print(model)

# Вывод параметров оптимизатора
print(optimizer)


