import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# Функция обучения модели с сохранением значения потерь
def train_model_with_loss(model, criterion, optimizer, X_train, y_train, X_test, y_test, num_epochs=100):
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(torch.Tensor(X_train))
        loss_train = criterion(outputs, torch.LongTensor(y_train))
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())

        with torch.no_grad():
            outputs = model(torch.Tensor(X_test))
            loss_test = criterion(outputs, torch.LongTensor(y_test))
            test_losses.append(loss_test.item())

    return train_losses, test_losses

# Создание и обучение модели
input_size = X.shape[1]
output_size = len(set(y))
hidden_size = 40
lr = 0.01

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

train_losses, test_losses = train_model_with_loss(model, criterion, optimizer, X_train, y_train, X_test, y_test)

# Построение графика потерь
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Test Losses')
plt.show()
