import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def runGridSearch(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

data = pd.read_csv(r"GeneratedDataSet\ModelDataSet.csv")
data.columns = data.columns.str.strip().str.replace(' ', '').str.lower()

# predire 'target' en fonction des 'features'
data_to_predict = "attrition"
target = data[data_to_predict]
Input_features = data.drop(data_to_predict, axis=1)

Training_data, Test_data, train_labels, test_labels = train_test_split(Input_features, target, test_size=0.2, random_state=42)

# debug
print("\n=== Données d'entraînement ===")
print(Training_data.head())
print("\n=== Données de test ===")
print(Test_data.head())
print("\n=== Labels d'entraînement ===")
print(train_labels.head())
print("\n=== Labels de test ===")
print(test_labels.head())

# Définir une grille de paramètres pour la recherche par grille
param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'positive': [True, False]
}

# Utiliser LinearRegression pour la recherche par grille
linear_regression = LinearRegression()

best_model = runGridSearch(linear_regression, param_grid, Training_data, train_labels)

y_pred = best_model.predict(Test_data)
mse = mean_squared_error(test_labels, y_pred)
r2 = r2_score(test_labels, y_pred)

print("\n=== Résultats de la régression linéaire ===")
print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
print(f"Coefficient de détermination (R²) : {r2:.2f}")


coefficients = pd.DataFrame({
    'Feature': Input_features.columns,
    'Coefficient': best_model.coef_
})
print("\n=== Coefficients du modèle ===")
print(coefficients.sort_values(by='Coefficient', ascending=False))

coefficients.to_csv("regression_coefficients.csv", index=False)

plt.figure(figsize=(10, 6))
# Calculer les erreurs absolues
# tableau de la taille de la plus petite liste
min_length = min(len(target), len(y_pred))
target = target[:min_length]
y_pred = y_pred[:min_length]
errors = target - y_pred

# Draw dashed lines at target min and max
plt.axhline(y=target.min(), color='lightgray', linestyle='--', linewidth=1, zorder=-1)
plt.axhline(y=target.max(), color='lightgray', linestyle='--', linewidth=1, zorder=-1)
plt.axvline(x=target.min(), color='lightgray', linestyle='--', linewidth=1, zorder=-1)
plt.axvline(x=target.max(), color='lightgray', linestyle='--', linewidth=1, zorder=-1)

# Ajuster les erreurs pour qu'elles soient dans la plage des valeurs cibles
adjusted_errors = np.where(y_pred < target.min(), 1, errors)
adjusted_errors = np.where(y_pred > target.max(), 1, adjusted_errors)

# normaliser les erreurs a [0, 1]
norm_errors = np.abs(adjusted_errors)

print("\n=== Erreurs ===")
for error, index in zip(errors, range(100)):
    print(f"expected: {target[index]} | predicted: {y_pred[index]:.2f} | error: {error:.2f} | normalized error: {norm_errors[index]:.2f}")

print("max error:", errors.max())
print("min error:", errors.min())

# [0, 1] -> [0, 1] (avec f(x) = x^0.5)
precision = np.power(norm_errors, 0.5)

cmap = LinearSegmentedColormap.from_list('precision_cmap', ['#5BAFFC', '#FD4F59'])
colors = cmap(precision)

plt.scatter(target, y_pred, alpha=0.75, c=colors, label='Predictions', marker='s')
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.title("Predictions vs. Actual Values")
plt.legend()
plt.axis('equal')
plt.xlim([target.min(), target.max()])
plt.ylim([min(y_pred.min(), target.min())-0.1, max(y_pred.max(), target.max())+0.1])
plt.show()