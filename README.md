# Employee Attrition Prediction

This project aims to predict employee attrition using various regression models. The dataset includes employee information, survey data, and in/out time records. The project involves data preprocessing, model training, and evaluation.

## Requirements

The required Python packages are listed in the `requirements.txt` file. You can install them using:
```sh
pip install -r
```

## Data Preprocessing

The data preprocessing script DataPreprocessing.py performs the following steps:

1. Load datasets from the DataSets/ directory.
2. Remove leading/trailing whitespaces from values and column names.
3. Replace 'NA' values with np.nan.
4. Merge datasets on EmployeeID.
5. Keep only the required columns.
6. Remove rows with missing data.
7. Identify numerical and non-numerical columns.
8. Normalize non-numerical columns.
9. Calculate average hours worked per day for each employee.
10. Normalize numerical columns.
11. Save the preprocessed data to GeneratedDataSet/ModelDataSet.csv.

## Model Training and Evaluation

The LinearRegression.py script performs the following steps:

1. Load the preprocessed data from GeneratedDataSet/ModelDataSet.csv.
2. Split the data into training and test sets.
3. Define parameter grids for various regression models (Linear Regression, Ridge, Lasso, ElasticNet).
4. Run grid search to find the best model and parameters.
5. Evaluate the best model using Mean Squared Error (MSE), R² score, and Mean Absolute Error (MAE).
6. Save the regression coefficients to regression_coefficients.csv.
7. Visualize the predictions vs. actual values

# Bibliography
## Documentation
- Course on linear regression : https://eric.univ-lyon2.fr/ricco/cours/slides/regularized_regression.pdf
A course to help you understand the basics of linear regression, and to discover other algorithms that can be used to refine the results obtained, such as Ridge or ElasticNet

- Scikit-learn documentation on the linear regression : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
the Scikit-learn library documentation for using linear regression, with a redirection to the pages for Ridge, lasso and ElasticNet

## Scientific article

- An elastic net regression model for predicting the risk of ICU admission and death for hospitalized patients with COVID-19, Wei Zou, Nature, June 2024 : https://www.nature.com/articles/s41598-024-64776-0
A scientific article describing the use of linear regression to predict the number of people with COVID-19 in intensive care or who die in hospitals


- Predicting high sensitivity C-reactive protein levels and their associations in a large population using decision tree and linear regression, Somayeh Ghiasi Hafezi, Nature, December 2024 : https://www.nature.com/articles/s41598-024-81714-2
Study designed to predict the sensitivity of C-reactive protein, which plays a major role in liver inflammation. This study uses linear regression and decision trees to feed their predictions