from math import ceil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

dataSetsPath = 'DataSets/'
modelDataSetsPath = 'GeneratedDataSet/'

# Define normalization range
normalizationRange = (-1, 1)

if normalizationRange[0] >= normalizationRange[1]:
    raise ValueError("Invalid normalization range. The minimum value must be less than the maximum value.")

print(f"Normalization range: {normalizationRange}")

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

# print the first 20 rows of the merged data
print(mergedData[['TotalWorkingYears']].head(20))

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

finalDataInitialRowCount = finalData.shape[0]

# Ensure Attrition is binary (0 for 'No' and 1 for 'Yes')
finalData['Attrition'] = finalData['Attrition'].map({'No': 0, 'Yes': 1})

# Step 5: Remove rows with any 'NA' or missing data
print("Removing rows with any 'NA' or missing data...")
initialRowCount = finalData.shape[0]
finalData.dropna(inplace=True)
finalRowCount = finalData.shape[0]
rowsRemoved = initialRowCount - finalRowCount
print(f"Number of rows removed due to missing data: {rowsRemoved} ({(rowsRemoved/initialRowCount)*100:.2f}%)")

# Step 6: Identify numerical and non-numerical columns
print("Identifying numerical and non-numerical columns...")
numericalColumns = []
nonNumericalColumns = []

for column in finalData.columns:
    if column == 'EmployeeID' or column == 'Attrition':
        continue
    
    # Get first value
    sample_value = finalData[column].iloc[0]
    try:
        float(sample_value)
        numericalColumns.append(column)
    except (ValueError, TypeError):
        nonNumericalColumns.append(column)

print("\nNumerical columns:", numericalColumns)
print("Non-numerical columns:", nonNumericalColumns)

# Print unique values and normalize non-numerical columns
print("\nNon-numerical columns unique values and mapping:")
mappingDict = {}
for column in nonNumericalColumns:
    unique_values = finalData[column].unique()
    n_values = len(unique_values)
    print(f"\n{column} - {n_values} unique values:")
    
    # Create mapping to normalized values between -1 and 1
    step = (normalizationRange[1] - normalizationRange[0]) / (n_values - 1) if n_values > 1 else 0
    value_map = {val: normalizationRange[0] + i * step for i, val in enumerate(sorted(unique_values))}
    
    # Print mapping
    for val, normalized in value_map.items():
        print(f"  {val}: {normalized:.2f}")
    
    # Apply mapping
    finalData[column] = finalData[column].map(value_map)
    
    # Add to mapping_dict
    mappingDict[column] = list(value_map.keys())

# Calculate average hours worked per day for each employee
print("\nCalculating average hours worked per day for each employee...")
datetimeFormat = '%Y-%m-%d %H:%M:%S'  # Specify the datetime format
for column in inTime.columns[1:]:
    inTime[column] = pd.to_datetime(inTime[column], format=datetimeFormat, errors='coerce')
    outTime[column] = pd.to_datetime(outTime[column], format=datetimeFormat, errors='coerce')

hoursWorked = outTime.iloc[:, 1:].subtract(inTime.iloc[:, 1:]).applymap(lambda x: x.total_seconds() / 3600 if pd.notnull(x) else np.nan)
averageHoursWorked = hoursWorked.mean(axis=1)
averageHoursWorked.name = 'AverageHoursWorked'

# Merge average hours worked with the final dataset
finalData = pd.merge(finalData, averageHoursWorked, left_on='EmployeeID', right_index=True, how='left')


# Add 'AverageHoursWorked' to numerical columns
numericalColumns.append('AverageHoursWorked')

for column in numericalColumns:
    finalData[column] = pd.to_numeric(finalData[column], errors='coerce')

# Normalize numerical columns
print("\nNumerical columns min-max values:")
for column in numericalColumns:
    maxValue = finalData[column].max()
    minValue = finalData[column].min()
    print(f"{column} - Min: {minValue}, Max: {maxValue}")
    mappingDict[column] = [minValue, maxValue]

# Apply MinMaxScaler to numerical columns
scaler = MinMaxScaler(feature_range=normalizationRange)
finalData[numericalColumns] = scaler.fit_transform(finalData[numericalColumns])

# Reorder mappingDict to match the order of columns in finalData
orderedMappingDict = {column: mappingDict[column] for column in finalData.columns if column in mappingDict}

# Save mapping values to a CSV file
mappingDf = pd.DataFrame([orderedMappingDict])
mappingDf.to_csv(modelDataSetsPath + 'MappingValues.csv', index=False)

# Final check for any missing data (and remove the row where 'AverageHoursWorked' is missing)
print("\nFinal check for missing data...")
initialRowCount = finalData.shape[0]
finalData.dropna(inplace=True)
finalRowCount = finalData.shape[0]
rowsRemoved = initialRowCount - finalRowCount
print(f"Number of rows removed due to missing data: {rowsRemoved} ({(rowsRemoved/initialRowCount)*100:.2f}%)")

# Print the removed percentage of rows
print(f"\nInitial row count: {finalDataInitialRowCount}")
print(f"Final row count: {finalRowCount}")
print(f"Percentage of rows removed: {(rowsRemoved/finalDataInitialRowCount)*100:.2f}%")

# Save the preprocessed data
print(f"\nSaving preprocessed data to {modelDataSetsPath}ModelDataSet.csv...")
finalData.to_csv(modelDataSetsPath + 'ModelDataSet.csv', index=False)
print("Done")