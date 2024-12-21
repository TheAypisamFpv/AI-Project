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

<br>

## Running the Deep Neural Network (from [NeuralNetworkVisualisation](https://github.com/TheAypisamFpv/NeuralNetworkVisualisation))
- The `RunNeuralNet.py` script initializes the neural network application.
- The application allows users to select a model file and visualize the neural network's predictions.
- The visualization includes neuron connections and output values, with colors indicating the neuron's output value.
- The application uses hierarchical clustering to cluster neurons when the number of neurons exceeds a threshold.

<br>

## Models Directory
The `Models\` directory contains the following files for each trained model (each model has its own subdirectory):
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.keras`: The trained neural network model file.
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.params`: The hyperparameters used for training the model.
- `Model_<train_accuracy>_<validation_accuracy>_<elapsed_time>.gif`: An animated plot showing the learning curves for accuracy and loss over epochs.
- `MappingValues.csv`: The mapping values used during preprocessing.
- `FeatureImportance.csv`: The feature importance values calculated using SHAP.
- `ValidationAccuracyHistory.png`: A plot showing the validation accuracy history during grid search.

<br>

## Bibliography
### Official Documentation
- #### [TensorFlow Documentation](https://www.tensorflow.org/guide) (Mar 2, 2023):
    Official guide provided by the TensorFlow team. It’s reliable because it’s maintained and updated by TensorFlow developers. It includes in-depth tutorials and API references, making it an essential resource for understanding neural network implementation and training.

- #### [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html) (Dec, 2024):
    Scikit-learn’s documentation is an authoritative source for machine learning algorithms, including those used in neural networks. It’s well-structured, frequently updated, and provides examples for practical applications.

### YouTube Videos
- #### [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (Oct 5, 2017 - Nov 20, 2024):
    Extensive playlist of 8 videos on neural networks and LLMs (covers Gradient descent, Backpropagation and Transformers (the ‘T’ in chat GPT)).

    Watch time. ~2h total

- #### [Sebastian Lague: How to Create a Neural Network](https://www.youtube.com/watch?v=hfMk-kjRv4c) (Aug 12, 2022):
    Sebastian demonstrates how to build and train a neural network to recognize doodles. He highlights the architecture, training process, and key optimizations to enhance accuracy in identifying drawings.

    Watch time : ~55 mins

- #### [Pezzza's Work: How to Train Simple AIs](https://www.youtube.com/watch?v=EvV5Qtp_fYg) (May 3, 2024):
    Pezzza demonstrates a simple method for training AIs with evolutionary neural networks using an inverted pendulum. It emphasizes the role of fitness functions and agent selection in enhancing performance and stability.

    Watch time : ~13 mins

### Artificial Intelligence Tools
- #### [GitHub Copilot](https://github.com/features/copilot) (latest training dataset date not provided by Microsoft):
    AI-based coding assistant. Helpful for writing code snippets, but its suggestions should always be carefully reviewed as it may produce errors.
Mainly used for debugging.

- #### [ChatGPT 4o](https://openai.com/chatgpt) (latest training dataset date : Oct. 2023, with internet access):
    Conversational AI tool useful for brainstorming and explaining concepts. While helpful for understanding neural networks, it is not always accurate or complete and should be used alongside verified sources.
