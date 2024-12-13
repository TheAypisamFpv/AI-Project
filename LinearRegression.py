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
errors = test_labels - y_pred

# Normaliser les erreurs pour qu'elles soient comprises entre 0 et 1
norm_errors = np.square((errors - errors.min()) / (errors.max() - errors.min()))

# Définir une fonction de mappage pour interpoler les couleurs
cmap = LinearSegmentedColormap.from_list('error_cmap', ['#5BAFFC', '#FD4F59'])
colors = cmap(norm_errors)

plt.scatter(test_labels, y_pred, c=colors, alpha=0.5, label='Prédictions', marker='s')

plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Prédictions vs. Valeurs réelles")
plt.legend()
plt.axis('equal')
plt.xlim([test_labels.min(), test_labels.max()])
plt.ylim([test_labels.min(), test_labels.max()])
plt.show()