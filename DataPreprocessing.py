import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set default font size for all plots
plt.rcParams.update({'font.size': 7})

# Step 1: Understand the Data
print("Loading datasets...")
generalData = pd.read_csv('DataSets/general_data.csv')
employeeSurveyData = pd.read_csv('DataSets/employee_survey_data.csv')
managerSurveyData = pd.read_csv('DataSets/manager_survey_data.csv')
inTime = pd.read_csv('DataSets/in_out_time/in_time.csv')
outTime = pd.read_csv('DataSets/in_out_time/out_time.csv')

print("Stripping leading/ending whitespaces from all values and column names...")
generalData.columns = generalData.columns.str.strip()
employeeSurveyData.columns = employeeSurveyData.columns.str.strip()
managerSurveyData.columns = managerSurveyData.columns.str.strip()
inTime.columns = inTime.columns.str.strip()
outTime.columns = outTime.columns.str.strip()

generalData = generalData.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
employeeSurveyData = employeeSurveyData.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
managerSurveyData = managerSurveyData.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

print("Handling missing values...")
employeeSurveyData.replace('NA', np.nan, inplace=True)
numericCols = employeeSurveyData.select_dtypes(include=[np.number]).columns
employeeSurveyData[numericCols] = employeeSurveyData[numericCols].fillna(employeeSurveyData[numericCols].mean())

# Rename the first column of inTime and outTime to EmployeeID
inTime.rename(columns={inTime.columns[0]: 'EmployeeID'}, inplace=True)
outTime.rename(columns={outTime.columns[0]: 'EmployeeID'}, inplace=True)

print("Replacing invalid datetime strings with NaN...")
inTime.replace('NA', np.nan, inplace=True)
outTime.replace('NA', np.nan, inplace=True)

print("Merging datasets on EmployeeID...")
mergedData = pd.merge(generalData, employeeSurveyData, on='EmployeeID', how='left')
mergedData = pd.merge(mergedData, managerSurveyData, on='EmployeeID', how='left')

print("Converting categorical variables into numerical values...")
categoricalColumns = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18']
mergedData = pd.get_dummies(mergedData, columns=categoricalColumns, drop_first=True)

print("Extracting useful features from IN_TIME and OUT_TIME...")
inTime.set_index('EmployeeID', inplace=True)
outTime.set_index('EmployeeID', inplace=True)

print("Converting time columns to datetime...")
inTime = inTime.apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', errors='coerce')
outTime = outTime.apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', errors='coerce')

print("Calculating total working hours for each employee...")
workingHours = (outTime - inTime).apply(lambda x: x.map(lambda y: y.total_seconds() / 3600 if pd.notnull(y) else np.nan))
workingHours['TotalWorkingHours'] = workingHours.sum(axis=1)

# Identify employees with 0 working hours
zeroWorkingHours = workingHours[workingHours['TotalWorkingHours'] == 0]
print(f"Number of employees with 0 working hours: {len(zeroWorkingHours)}")

# Handle employees with 0 working hours (e.g., remove them from the dataset)
mergedData = mergedData[~mergedData['EmployeeID'].isin(zeroWorkingHours.index)]

print("Adding total working hours to merged data...")
mergedData = pd.merge(mergedData, workingHours[['TotalWorkingHours']], left_on='EmployeeID', right_index=True, how='left')

print("Handling remaining missing values in merged data...")
mergedData.replace('NA', np.nan, inplace=True)
numericCols = mergedData.select_dtypes(include=[np.number]).columns
mergedData[numericCols] = mergedData[numericCols].fillna(mergedData[numericCols].mean())

# Step 3: Exploratory Visualization
print("Analyzing correlations...")
correlationMatrix = mergedData.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlationMatrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print("Visualizing distributions...")
# Histogram of Total Working Hours
plt.figure(figsize=(10, 6))
plt.hist(mergedData['TotalWorkingHours'].dropna(), bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Total Working Hours')
plt.xlabel('Total Working Hours')
plt.ylabel('Number of Employees')
plt.grid(True)
plt.show()

# Box plot of Monthly Income by Attrition
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition_Yes', y='MonthlyIncome', data=mergedData)
plt.title('Monthly Income by Attrition')
plt.xlabel('Attrition')
plt.ylabel('Monthly Income')
plt.show()

# Box plot of Age by Attrition
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition_Yes', y='Age', data=mergedData)
plt.title('Age by Attrition')
plt.xlabel('Attrition')
plt.ylabel('Age')
plt.show()

print("Done.")