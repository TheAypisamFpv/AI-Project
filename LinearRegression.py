import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("GeneratedDataSet\ModelDataSet.csv")

required_columns = [
    'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
    'EducationField', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus',
    'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'JobInvolvement',
    'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction',
    'WorkLifeBalance', 'AverageHoursWorked'
]

missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Les colonnes suivantes sont manquantes dans le dataset : {missing_columns}")

# Séparer les features (X) et la target (y)
y = data['Turnover']
X = data.drop(columns=[required_columns])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== Résultats de la régression linéaire ===")
print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
print(f"Coefficient de détermination (R²) : {r2:.2f}")

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\n=== Coefficients du modèle ===")
print(coefficients.sort_values(by='Coefficient', ascending=False))


coefficients.to_csv("regression_coefficients.csv", index=False)


plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Prédictions vs. Valeurs réelles")
plt.show()