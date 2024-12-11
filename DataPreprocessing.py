import pandas as pd
import numpy as np

dataSetsPath = 'DataSets/'
modelDataSetsPath = 'GeneratedDataSet/'

# Step 1: Load datasets
print("Loading datasets...")
generalData = pd.read_csv(dataSetsPath + 'general_data.csv')
employeeSurveyData = pd.read_csv(dataSetsPath + 'employee_survey_data.csv')
managerSurveyData = pd.read_csv(dataSetsPath + 'manager_survey_data.csv')
inTime = pd.read_csv(dataSetsPath + 'in_out_time/in_time.csv')
outTime = pd.read_csv(dataSetsPath + 'in_out_time/out_time.csv')

# Step 2: Remove any leading/ending whitespaces for each value
print("Stripping leading/ending whitespaces from all values and column names...")
generalData.columns = generalData.columns.str.strip()
employeeSurveyData.columns = employeeSurveyData.columns.str.strip()
managerSurveyData.columns = managerSurveyData.columns.str.strip()
inTime.columns = inTime.columns.str.strip()
outTime.columns = outTime.columns.str.strip()

generalData = generalData.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
employeeSurveyData = employeeSurveyData.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
managerSurveyData = managerSurveyData.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Replace 'NA' with np.nan
generalData.replace('NA', np.nan, inplace=True)
employeeSurveyData.replace('NA', np.nan, inplace=True)
managerSurveyData.replace('NA', np.nan, inplace=True)
inTime.replace('NA', np.nan, inplace=True)
outTime.replace('NA', np.nan, inplace=True)

# Rename the first column of inTime and outTime to EmployeeID
inTime.rename(columns={inTime.columns[0]: 'EmployeeID'}, inplace=True)
outTime.rename(columns={outTime.columns[0]: 'EmployeeID'}, inplace=True)

# Step 3: Merge all datasets using EmployeeID
print("Merging datasets on EmployeeID...")
mergedData = pd.merge(generalData, employeeSurveyData, on='EmployeeID', how='left')
mergedData = pd.merge(mergedData, managerSurveyData, on='EmployeeID', how='left')
mergedData = pd.merge(mergedData, inTime, on='EmployeeID', how='left')
mergedData = pd.merge(mergedData, outTime, on='EmployeeID', how='left')

# Step 4: Keep only the required columns
requiredColumns = [
    'EmployeeID', 'Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus',
    'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction',
    'JobSatisfaction', 'WorkLifeBalance'
]
finalData = mergedData[requiredColumns].copy()

# Step 5: Remove rows with any 'NA' or missing data
print("Removing rows with any 'NA' or missing data...")
initialRowCount = finalData.shape[0]
finalData.dropna(inplace=True)
finalRowCount = finalData.shape[0]
rowsRemoved = initialRowCount - finalRowCount
print(f"Number of rows removed due to missing data: {rowsRemoved} ({(rowsRemoved/initialRowCount)*100:.2f}%)")

# Step 6: Save the new CSV
print("Saving the preprocessed data to ModelDataSet.csv...")
finalData.to_csv(modelDataSetsPath + 'ModelDataSet.csv', index=False)

print("Done.")