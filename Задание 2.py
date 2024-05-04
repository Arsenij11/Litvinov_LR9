import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка данных
digits = load_digits()
X = digits.data
y = digits.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Функция обучения и оценки модели
def train_and_evaluate_model(input_size, hidden_size, output_size, num_layers, activation):
    model = nn.Sequential()
    model.add_module("input_layer", nn.Linear(input_size, hidden_size))
    if activation == "relu":
        model.add_module("activation", nn.ReLU())
    elif activation == "sigmoid":
        model.add_module("activation", nn.Sigmoid())
    for _ in range(num_layers - 1):
        model.add_module(f"hidden_layer{_}", nn.Linear(hidden_size, hidden_size))
        if activation == "relu":
            model.add_module(f"activation{_}", nn.ReLU())
        elif activation == "sigmoid":
            model.add_module(f"activation{_}", nn.Sigmoid())
    model.add_module("output_layer", nn.Linear(hidden_size, output_size))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Обучение модели
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(torch.Tensor(X_train))
        loss = criterion(outputs, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()

    # Оценка точности модели
    with torch.no_grad():
        outputs = model(torch.Tensor(X_test))
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted.numpy())

    return accuracy

# Параметры для исследования
input_size = X.shape[1]
output_size = len(set(y))
num_layers_list = [1, 2, 3]  # Различные количества слоев
hidden_size_list = [20, 40, 60]  # Различное количество нейронов в скрытом слое
activation_functions = ["relu", "sigmoid"]  # Различные методы активации

# Проведение исследования
for num_layers in num_layers_list:
    for hidden_size in hidden_size_list:
        for activation in activation_functions:
            accuracy = train_and_evaluate_model(input_size, hidden_size, output_size, num_layers, activation)
            print(f"Layers: {num_layers}, Hidden Size: {hidden_size}, Activation: {activation}, Accuracy: {accuracy}")
