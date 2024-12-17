import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
data = pd.read_csv("GeneratedDataSet/ModelDataSet.csv")

# Séparation des caractéristiques (X) et de la variable cible (y)
X = data.drop(["EmployeeID", "Attrition"], axis=1)
y = data["Attrition"]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création et entraînement du modèle SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Prédictions sur l'ensemble de test
y_pred = svm_model.predict(X_test_scaled)

# Évaluation du modèle
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Calcul de la précision du modèle
accuracy = svm_model.score(X_test_scaled, y_test)
print(f"\nPrécision du modèle : {accuracy:.2f}")




# Charger les données
#data = pd.read_csv("ModelDataSet.csv")

# Calculer la matrice de corrélation
corr_matrix = data.corr()

# Créer la heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matrice de corrélation des variables")
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(data=data, x="Age", kde=True, ax=axes[0, 0])
sns.histplot(data=data, x="MonthlyIncome", kde=True, ax=axes[0, 1])
sns.histplot(data=data, x="TotalWorkingYears", kde=True, ax=axes[1, 0])
sns.histplot(data=data, x="YearsAtCompany", kde=True, ax=axes[1, 1])

plt.tight_layout()
plt.show()


attrition_by_dept = data.groupby("Department")["Attrition"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=attrition_by_dept.index, y=attrition_by_dept.values)
plt.title("Taux d'attrition par département")
plt.ylabel("Taux d'attrition")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="Age", y="MonthlyIncome", hue="Attrition")
plt.title("Relation entre l'âge et le revenu mensuel")
plt.show()
