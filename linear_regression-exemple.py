import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regr.predict(diabetes_X_test)

print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(diabetes_X_test, diabetes_y_test, color="black", label="Dados Reais")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=2, label="Linha de Regressão")

plt.title("Regressão Linear - Predição de Diabetes")
plt.xlabel("Índice de Massa Corporal (IMC)")
plt.ylabel("Progressão da Diabetes (medida em anos)")
plt.legend()

plt.grid()

plt.text(0.05, 0.95, f'Coeficiente: {regr.coef_[0]:.2f}\n'
                     f'MSE: {mean_squared_error(diabetes_y_test, diabetes_y_pred):.2f}\n'
                     f'R²: {r2_score(diabetes_y_test, diabetes_y_pred):.2f}', 
         fontsize=12, transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

plt.show()
