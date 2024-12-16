from sklearn.model_selection import ParameterGrid, train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.exceptions import ConvergenceWarning
import warnings

def runGridSearch(models_param_grids, X_train, y_train):
    best_score = float('inf')
    best_params = None
    best_model = None
    best_model_name = None

    for model_name, (model_class, param_grid) in models_param_grids.items():
        for params in ParameterGrid(param_grid):
            model = model_class(**params)
            try:
                # Utiliser la validation croisée pour évaluer le modèle
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                score = -scores.mean()  # Convertir en MSE positif
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    best_model_name = model_name
            except Exception as e:
                print(f"Error with model {model_name} and parameters {params}: {e}")

    print(f"Best model: {best_model_name} with parameters: {best_params}")
    best_model.fit(X_train, y_train)  # Entraîner le meilleur modèle sur toutes les données d'entraînement
    return best_model

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

# Définir une grille de paramètres pour chaque modèle
models_param_grids = {
    'LinearRegression': (LinearRegression, {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True, False]
    }),
    'Ridge': (Ridge, {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'max_iter': [1000, 5000, 10000]
    }),
    'Lasso': (Lasso, {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 5000, 10000]
    }),
    'ElasticNet': (ElasticNet, {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0.1, 0.5, 0.9],
        'fit_intercept': [True, False],
        'max_iter': [1000, 5000, 10000]
    })
}

# Exécuter la recherche par grille
best_model = runGridSearch(models_param_grids, Training_data, train_labels)

y_pred = best_model.predict(Test_data)
mse = mean_squared_error(test_labels, y_pred)
r2 = r2_score(test_labels, y_pred)
mae = mean_absolute_error(test_labels, y_pred)

print("\n=== Résultats de la régression linéaire ===")
print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
print(f"Coefficient de détermination (R²) : {r2:.2f}")
print(f"Erreur absolue moyenne (MAE) : {mae:.2f}")

coefficients = pd.DataFrame({
    'Feature': Input_features.columns,
    'Coefficient': best_model.coef_
})
print("\n=== Coefficients du modèle ===")
print(coefficients.sort_values(by='Coefficient', ascending=False))

coefficients.to_csv("regression_coefficients.csv", index=False)

# Afficher les 20 premières valeurs "à prédire" et celles finalement prédites
print("\n=== Comparaison des valeurs réelles et prédites ===")
comparison_df = pd.DataFrame({
    'Valeurs réelles': test_labels[:20].values,
    'Valeurs prédites': y_pred[:20]
})
print(comparison_df)

# Visualisation des prédictions vs. valeurs réelles
plt.figure(figsize=(10, 6))

# Calculer les erreurs absolues
errors = test_labels - y_pred

# Normaliser les erreurs pour qu'elles soient comprises entre 0 et 1
norm_errors = np.square((errors - errors.min()) / (errors.max() - errors.min()))

# Définir une fonction de mappage pour interpoler les couleurs
cmap = LinearSegmentedColormap.from_list('error_cmap', ['#5BAFFC', '#FD4F59'])
colors = cmap(norm_errors)

# Utiliser des boîtes pour visualiser les prédictions
plt.scatter(test_labels, y_pred, c=colors, alpha=0.5, label='Prédictions', marker='s')

plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Prédictions vs. Valeurs réelles")
plt.legend()
plt.axis('equal')
plt.xlim([test_labels.min(), test_labels.max()])
plt.ylim([test_labels.min(), test_labels.max()])
plt.show()