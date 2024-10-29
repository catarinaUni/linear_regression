import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def regressao_linear(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    b1 = numerator / denominator
    
    b0 = y_mean - b1 * X_mean
    
    return b0, b1

X = np.array([0, 7, 10, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 1, 2, 8, 10, 12, 14, 16, 18, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

b0, b1 = regressao_linear(X_train.flatten(), y_train)

y_pred = b0 + b1 * X_test.flatten()

r_squared = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
print(f"b0 (intercepto): {b0}, b1 (inclinação): {b1}")
print(f"R²: {r_squared}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, b0 + b1 * X.flatten(), color='red', label='Linha de Regressão', linewidth=2)
plt.title('Regressão Linear Simples')
plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Dados reais')
plt.title('Pontos de Dados Reais')
plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.legend()

plt.tight_layout()
plt.show()
