
# perform hyperparameter tuning with gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd 

# load data

file_path = 'GeneratedDataSet/ModelDataSet.csv'

df = pd.read_csv(file_path)

X = df.drop('Attrition ', axis=1)
y = df['Attrition ']

# class distribution to check if it is imbalanced or not
if y.value_counts()[0] > y.value_counts()[1]:
    print("There is an imbalance:", y.value_counts())

    # oversample the minority class
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    # check class distribution again
    print("After oversampling:", y.value_counts())
else:   
    print("There is no imbalance:", y.value_counts())

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# instantiate model
model = LogisticRegression(penalty='l2', solver='liblinear')

# param grid
param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2'],  # Add L1 regularization
}


# instantiate gridsearchn
grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

# print best params
print("Best hyperparameters",grid.best_params_)
print("Best score attained",grid.best_score_)
print("Best model",grid.best_estimator_)

# print accuracy
print(grid.score(X_test, y_test))

# get best model
best_model = grid.best_estimator_

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict on the test set
y_pred = best_model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
disp.plot(cmap=plt.cm.Blues)

# Add title and labels
plt.title("Confusion Matrix")
plt.show()
plt.savefig('Plots/LogisticRegressionCM.png')


# print classification report
from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(y_test, y_pred))

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# Assuming you have a trained model and test data
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
RocCurveDisplay.from_predictions(y_test, y_proba)

# Add title and labels
plt.title("ROC Curve")
plt.show()
 # save plot as image
plt.savefig('Plots/LogisticRegressionROC.png')

# Visualize features influencing the model
# Get feature coefficients
feature_coefficients = best_model.coef_[0]  # Logistic Regression has a 2D coef_ array
feature_names = X_train.columns

# Create a DataFrame to display coefficients
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": feature_coefficients
}).sort_values(by="Coefficient", ascending=False)

# Display the coefficients
print("Feature coefficients")
print(coef_df)

# Plot the feature coefficients
plt.figure(figsize=(10, 10))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()


# save model
import joblib
joblib.dump(grid, 'Models/LogisticRegression.pkl')