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



## Deep Neural Network Training
- Split the dataset into training and testing sets with stratification.
- Compute class weights to handle class imbalance.
- Build the deep neural network (DNN) model with specified architecture and parameters.
- Train the model with early stopping based on validation loss.
- Evaluate the model on the test set.
- Perform additional evaluation using SHAP values to determine feature importance.



## Best Models as of Now:
### With `AverageHoursWorked` (Validation Accuracy: 98%):

`TrainedModel_333596_3`

### Parameters:
```json
{
    "batchSize": 32,
    "dropoutRate": 0.3,
    "epochs": 100,
    "hiddenActivation": "relu",
    "inputActivation": "relu",
    "l2_reg": 0.001,
    "layers": [25, 256, 128, 64, 1],
    "learningRate": 0.0005,
    "loss": "binary_crossentropy",
    "metrics": ["Accuracy", "Precision"],
    "optimizer": "adam",
    "outputActivation": "sigmoid",
    "randomSeed": 404
}
```

### Without `AverageHoursWorked` (Validation Accuracy: 97%):

I also trained a model without the `AverageHoursWorked` feature, in case the feature is not easily obtainable.

`TrainedModel_285607_14`

### Parameters:
```json
{
    "batchSize": 32,
    "dropoutRate": [0.5, 0.4, 0.3, 0.2, 0.1],
    "epochs": 100,
    "hiddenActivation": "relu",
    "inputActivation": "relu",
    "l2_reg": 0.001,
    "layers": [24, 512, 256, 128, 64, 1],
    "learningRate": 0.001,
    "loss": "binary_crossentropy",
    "metrics": ["Accuracy", "Precision"],
    "optimizer": "adam",
    "outputActivation": "sigmoid",
    "trainingTestingSplit": 0.2,
    "randomSeed": 404
}
```

These models can be found in the `models` directory in the folder of the same name.



## Running the Deep Neural Network
- The `RunNeuralNet.py` script initializes the neural network application.
- The application allows users to select a model file and visualize the neural network's predictions.
- The visualization includes neuron connections and output values, with colors indicating the neuron's output value.
- The application uses hierarchical clustering to cluster neurons when the number of neurons exceeds a threshold.



## Models Directory
The `Models\` directory contains the following files for each trained model (each model has its own subdirectory):
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.keras`: The trained neural network model file.
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.params`: The hyperparameters used for training the model.
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.gif`: An animated plot showing the learning curves for accuracy and loss over epochs.
- `MappingValues.csv`: The mapping values used during preprocessing.
- `FeatureImportance.csv`: The feature importance values calculated using SHAP.
- `ValidationAccuracyHistory.png`: A plot showing the validation accuracy history during grid search.



## Bibliography
### Documentations
- TensorFlow Documentation: https://www.tensorflow.org/guide
- Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
- SHAP Documentation: https://shap.readthedocs.io/en/latest/

### Youtube Videos:
-  [Neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) from 3Blue1Brown (video 1 to 4 but i watched everything)
- [How to Create a Neural Network (and Train it to Identify Doodles)](https://www.youtube.com/watch?v=hfMk-kjRv4c) from Sebastian Lague
- [How to train simple AIs](https://www.youtube.com/watch?v=EvV5Qtp_fYg) from Pezzza's Work

### AIs:
- Github Copilot