# AI Project

## Neural Network Branch

### Big Picture
- Sanitize dataset
- Separate dataset into training and testing
- Train model on training dataset
- Test model on testing dataset
- Visualize/validate model

---

### Model
- Deep Neural Network (DNN)

---

### Model Visualization
- The visualization shows the neural network's structure, with neurons displaying their output values.
- Neuron connections are colored based on the neuron's output value.
- The application uses hierarchical clustering to cluster neurons when the number of neurons exceeds a threshold.

---

### Requirements
#### Python 3.11.x - 3.12.x

Python can be downloaded from the official website: [python.org](https://www.python.org/downloads/)

- Create a virtual environment using the installed Python version
- Install the required packages using the following command:

```bash
pip install -r requirements.txt
```


## Data Preprocessing
- Load datasets from `DataSets/` directory.
- Remove leading/trailing whitespaces from all values and column names.
- Replace 'NA' with `np.nan`.
- Merge datasets on `EmployeeID`.
- Keep only the required columns.
- Ensure `Attrition` is binary (0 for 'No' and 1 for 'Yes').
- Remove rows with any 'NA' or missing data.
- Identify numerical and non-numerical columns.
- Normalize non-numerical columns to values between -1 and 1.
- Calculate average hours worked per day for each employee.
- Normalize numerical columns with padding for large values.
- Save the preprocessed data to `GeneratedDataSet/ModelDataSet.csv`.

---

### Deep Neural Network Training
- Split the dataset into training and testing sets with stratification.
- Compute class weights to handle class imbalance.
- Build the deep neural network (DNN) model with specified architecture and parameters.
- Train the model with early stopping based on validation loss.
- Evaluate the model on the test set.
- Perform additional evaluation using SHAP values to determine feature importance.

---

### Best Models as of Now:
#### With `AverageHoursWorked` (Validation Accuracy: 97%):

`TrainedModel_[25, 256, 128, 64, 1]_100_32_0.3_0.001_relu_relu_sigmoid_['Accuracy', 'Recall', 'Precision']_binary_crossentropy_adam(0.001)_0.2`

#### Without `AverageHoursWorked` (Validation Accuracy: 97%):

`TrainedModel_[24, 256, 128, 64, 1]_100_32_0.3_0.001_relu_relu_sigmoid_['Accuracy', 'Precision']_binary_crossentropy_adam(0.0005)_0.2`

These models can be found in the `models` directory in the folder of the same name.

---

### Running the Deep Neural Network
- The `RunNeuralNet.py` script initializes the neural network application.
- The application allows users to select a model file and visualize the neural network's predictions.
- The visualization includes neuron connections and output values, with colors indicating the neuron's output value.
- The application uses hierarchical clustering to cluster neurons when the number of neurons exceeds a threshold.

---

### Models Directory
The `Models\` directory contains the following files for each trained model (each model has its own subdirectory):
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.keras`: The trained neural network model file.
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.params`: The hyperparameters used for training the model.
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.gif`: An animated plot showing the learning curves for accuracy and loss over epochs.
- `MappingValues.csv`: The mapping values used during preprocessing.
- `FeatureImportance.csv`: The feature importance values calculated using SHAP.
- `ValidationAccuracyHistory.png`: A plot showing the validation accuracy history during grid search.