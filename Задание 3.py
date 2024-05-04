import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

# Загрузка данных
digits = load_digits()
X = digits.data
y = digits.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Проверяем доступность GPU и используем его, если возможно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Функция обучения модели с замером времени
def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=100):
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(torch.Tensor(X_train))
        loss = criterion(outputs, torch.LongTensor(y_train))
        loss.backward()
        optimizer.step()
    end_time = time.time()
    return end_time - start_time

# Параметры модели
input_size = X.shape[1]
output_size = len(set(y))
hidden_size = 40
lr = 0.01

# Создание модели
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Обучение модели на CPU
device = torch.device("cpu")
model = model.to(device)
cpu_time = train_model(model, criterion, optimizer, X_train, y_train)

# Обучение модели на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
gpu_time = train_model(model, criterion, optimizer, X_train, y_train)

print(f"Время обучения на CPU: {cpu_time} секунд")
print(f"Время обучения на GPU: {gpu_time} секунд")

# Вычисление ускорения
speedup = cpu_time / gpu_time
print(f"Ускорение: {speedup:.2f}x")
