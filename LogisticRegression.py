
# Load libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd 

# load data
file_path = 'GeneratedDataSet/ModelDataSet.csv'

df = pd.read_csv(file_path)

# drop the 'EmployeeID ' column for ethical reasons(privacy)
df.drop('EmployeeID ', axis=1, inplace=True)

df_copy = df.copy()

# find list of columns that are highly correlated
"""
Logistic Regression assumes that the features are independent of each other.It doesn;t work well when features are higlighly corelated with each other.
In order to improve the performance of the model. we must observe the correlation between features and remove highly correlated features.
"""
threshold = 0.7
correlated_features = set()
columns = df.columns
for i, col in enumerate(columns):
    for col2 in columns[i + 1:]:  # Compare only unique pairs
        corr = df[col].corr(df[col2])
        if abs(corr) > threshold:
            correlated_features.add((col, col2))
            print(f"Highly Correlated: {col} and {col2} -> Correlation: {corr}")


# Instead of dropping the columns, we can create a new column as a ratio of the two columns to preserve the information
for col, col2 in correlated_features:
    # Safely create a new column as a ratio
    new_col_name = f"{col.strip()}_per_{col2.strip()}"
    df_copy[new_col_name] = df_copy[col] / (df_copy[col2] + 1e-9)  
    # drop the original columns
    df_copy.drop([col, col2], axis=1, inplace=True)

# display new dataset for additional features
df = df_copy
print(df_copy.head())

# check correlation with 'Attrition'
"""
Logistic regression works well when the features are correlated with the target variable.
It is important to check for this correlation to see if the model will perform well or not.
"""
correlation = df.corr()['Attrition '].sort_values(ascending=False)
print('Correlation with Attrition') 
print(correlation)

"""
The correlation values are very low. This means that the features well correlated with our target variable.
Hence the model may not perform well.
"""

# Start building the model

# Split the data into X and y
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
    'penalty': ['l1', 'l2'],  
}


# instantiate gridsearch
grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

# print best params
print("Best hyperparameters:",grid.best_params_)
print("Best score attained:",grid.best_score_)
print("Best model:",grid.best_estimator_)

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
plt.savefig('Plots/LogisticRegressionCM.png')
plt.show()


# print classification report
from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(y_test, y_pred))

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# Get the probabilities for the positive class
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
RocCurveDisplay.from_predictions(y_test, y_proba)

# Add title and labels
plt.title("ROC Curve")
 # save plot as image
plt.savefig('Plots/LogisticRegressionROC.png')
plt.show()


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
plt.savefig('Plots/FeatureImportancePlot.png')
plt.show()

# save model
import joblib
joblib.dump(grid, 'Models/LogisticRegression.pkl')