
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
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)

# print accuracy
print(grid.score(X_test, y_test))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict on the test set
y_pred = grid.best_estimator_.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
disp.plot(cmap=plt.cm.Blues)

# Add title and labels
plt.title("Confusion Matrix")
plt.show()

# save model
import joblib
joblib.dump(grid, 'Models/LogisticRegression.pkl')
