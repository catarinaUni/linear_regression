# Importando bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

url = './Salary_Data.csv'
salary_data = pd.read_csv(url)

print(salary_data.head())

print(salary_data.isnull().sum())

salary_data = salary_data.dropna()

X = salary_data[['Years of Experience']]
y = salary_data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

plt.figure(figsize=(10, 6))

plt.scatter(salary_data['Years of Experience'], salary_data['Salary'], color='blue', label='Dados reais', alpha=0.5)

X_range = pd.DataFrame({'Years of Experience': range(int(salary_data['Years of Experience'].min()), 
                                                       int(salary_data['Years of Experience'].max()) + 1)})
y_range = model.predict(X_range)

plt.plot(X_range, y_range, color='red', label='Linha de Regressão', linewidth=2)

plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.title('Salário em Função dos Anos de Experiência')
plt.legend()
plt.grid()

plt.show()
