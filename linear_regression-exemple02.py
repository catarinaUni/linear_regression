import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = np.array([0, 7, 10, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 1, 2, 8, 10, 12, 14, 16, 18, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

r2_score = modelo.score(X_test, y_test)
print(f"R²: {r2_score}")

plt.scatter(X, y, color='blue')
plt.plot(X, modelo.predict(X), color='red')
plt.title('Regressão Linear Simples')
plt.xlabel('Variável Independente')
plt.ylabel('Variável Dependente')
plt.show()