import os
import json
import pandas as pd
import numpy as np
from math import floor
from progressBar import getProgressBar

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# remove tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore

import shap

from joblib import Parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set TensorFlow threading parameters
tf.config.threading.set_intra_op_parallelism_threads(11)
tf.config.threading.set_inter_op_parallelism_threads(11)

# Set environment variables to limit the number of threads used by OpenMP and other libraries
os.environ['OMP_NUM_THREADS'] = '11'
os.environ['OPENBLAS_NUM_THREADS'] = '11'
os.environ['MKL_NUM_THREADS'] = '11'
os.environ['VECLIB_MAXIMUM_THREADS'] = '11'
os.environ['NUMEXPR_NUM_THREADS'] = '11'

# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def saveCustomMapping(defaultMappingFilePath:str, updatedMappingFilePath:str, specifiedColumnsToDrop:list = []):
    """
    Load default mapping file and save the updated mapping file
    """
    # Load the default mapping file
    defaultMapping = pd.read_csv(defaultMappingFilePath)

    # Drop specified columns
    for column in specifiedColumnsToDrop:
        if column in defaultMapping.columns:
            defaultMapping = defaultMapping.drop(column, axis=1)

    # Save the updated mapping file
    if not updatedMappingFilePath.endswith('.csv'):
        updatedMappingFilePath += '.csv'
    
    defaultMapping.to_csv(updatedMappingFilePath, index=False)
    print(f"\nUpdated mapping file saved as '{updatedMappingFilePath}'\n")

def loadAndPreprocessData(filePath:str, specifiedColumnsToDrop:list = []):
    """Load and preprocess the dataset from a CSV file.

    This function reads data from the specified CSV file, performs preprocessing steps such as
    stripping whitespace from column names, dropping unnecessary columns, handling missing values,
    and encoding categorical variables. It then separates the features from the target variable.

    Args:
        filePath (str): The path to the CSV file containing the dataset.

    Returns:
        tuple: A tuple containing:
            - features (pd.DataFrame): The preprocessed feature columns.
            - target (pd.Series): The target variable column.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        KeyError: If the 'EmployeeID' or 'Attrition' columns are not found in the dataset.
    """
    # Load the dataset
    data = pd.read_csv(filePath)

    # Strip any leading or trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Drop EmployeeID as it is not a feature
    data = data.drop('EmployeeID', axis=1)

    # Drop specified columns
    for column in specifiedColumnsToDrop:
        print("Dropping columns :")
        if column in data.columns:
            data = data.drop(column, axis=1)
            print(f"\t-{column}")

    print()


    # Handle missing values by dropping rows with any NaN values
    data = data.dropna()

    # Encode categorical variables using LabelEncoder
    labelEncoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        labelEncoders[column] = LabelEncoder()
        data[column] = labelEncoders[column].fit_transform(data[column])

    # Separate features and target variable
    features = data.drop('Attrition', axis=1)
    target = data['Attrition']

    # Print column names for features and target
    print("Feature columns:", features.columns.tolist())
    print("Target column:", target.name)

    return features, target


def buildNeuralNetModel(
    layers: list[int],
    inputActivation: str,
    hiddenActivation: str,
    outputActivation: str,
    metrics: list,
    loss: str,
    optimizer: str,
    dropoutRate: float = 0.5,
    l2_reg: float = 0.01
):
    """Construct a Sequential neural network model based on the specified architecture and parameters.

    This function builds a neural network with an input layer, multiple hidden layers with dropout
    and L2 regularization, and an output layer. The model is then compiled with the given loss function,
    optimizer, and evaluation metrics.

    Args:
        layers (list[int]): A list specifying the number of neurons in each layer, including input and output layers.
        inputActivation (str): Activation function for the input layer.
        hiddenActivation (str): Activation function for the hidden layers.
        outputActivation (str): Activation function for the output layer.
        metrics (list): List of metrics to evaluate during training.
        loss (str): Loss function to optimize.
        optimizer (str): Optimizer to use for training.
        dropoutRate (float, optional): Dropout rate for dropout layers to prevent overfitting. Defaults to 0.5.
        l2_reg (float, optional): L2 regularization factor to apply to Dense layers. Defaults to 0.01.

    Returns:
        tf.keras.Model: The compiled Sequential neural network model.
    """
    model = Sequential()

    # Add input layer with specified activation and L2 regularization
    model.add(tf.keras.layers.Input(shape=(layers[0],)))
    model.add(Dense(layers[1], activation=inputActivation, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropoutRate))

    # Add hidden layers with specified activation, dropout, and L2 regularization
    for neurons in layers[2:-1]:
        model.add(Dense(neurons, activation=hiddenActivation, kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropoutRate))

    # Add output layer with specified activation and L2 regularization
    model.add(Dense(layers[-1], activation=outputActivation, kernel_regularizer=l2(l2_reg)))

    # Compile the model with specified loss, optimizer, and metrics
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def trainNeuralNet(
    features,
    target,
    layers: list[int],
    epochs: int,
    batchSize: int,
    inputActivation: str = 'relu',
    hiddenActivation: str = 'relu',
    outputActivation: str = 'sigmoid',
    metrics: list = ['Accuracy'],
    loss: str = 'binary_crossentropy',
    optimizer: str = 'adam',
    dropoutRate: float = 0.5,
    trainingTestingSplit: float = 0.2,
    l2_reg: float = 0.01,
    verbose: int = 1
):
    """Train the neural network model on the provided dataset.

    This function splits the dataset into training and testing sets, computes class weights to handle
    class imbalance, builds the neural network model, and trains it with early stopping based on validation loss.
    After training, it evaluates the model on the test set and performs additional evaluations.

    Args:
        features (pd.DataFrame): The preprocessed feature columns.
        target (pd.Series): The target variable column.
        layers (list[int]): The architecture of the neural network, specifying the number of neurons in each layer.
        epochs (int): Number of epochs to train the model.
        batchSize (int): Number of samples per gradient update.
        inputActivation (str, optional): Activation function for the input layer. Defaults to 'relu'.
        hiddenActivation (str, optional): Activation function for the hidden layers. Defaults to 'relu'.
        outputActivation (str, optional): Activation function for the output layer. Defaults to 'sigmoid'.
        metrics (list, optional): List of metrics to evaluate during training. Defaults to ['Accuracy'].
        loss (str, optional): Loss function to optimize. Defaults to 'binary_crossentropy'.
        optimizer (str, optional): Optimizer to use for training. Defaults to 'adam'.
        dropoutRate (float, optional): Dropout rate for dropout layers to prevent overfitting. Defaults to 0.5.
        trainingTestingSplit (float, optional): Fraction of the dataset to include in the test split. Defaults to 0.2.
        l2_reg (float, optional): L2 regularization factor to apply to Dense layers. Defaults to 0.01.
        verbose (int, optional): Verbosity mode (0 = silent, 1 = progress bar). Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - model (tf.keras.Model): The trained neural network model.
            - history (tf.keras.callbacks.History): A record of training loss values and metrics.

    Raises:
        ValueError: If the training or validation metrics are not available.
    """
    # Split the data into training and testing sets with stratification
    TrainingFeatures, TestFeatures, trainingLabels, testLabels = train_test_split(
        features,
        target,
        test_size=trainingTestingSplit,
        random_state=RANDOM_SEED,
        stratify=target
    )

    # Reset index of training labels
    trainingLabels = trainingLabels.reset_index(drop=True)

    # Compute class weights to handle class imbalance
    classes = np.unique(trainingLabels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=trainingLabels)
    classWeights = {cls: weight for cls, weight in zip(classes, class_weights)}
    if verbose: print(f"Computed class weights: {classWeights}")

    # Build the neural network model
    model = buildNeuralNetModel(
        layers=layers,
        inputActivation=inputActivation,
        hiddenActivation=hiddenActivation,
        outputActivation=outputActivation,
        metrics=metrics,
        loss=loss,
        optimizer=optimizer,
        dropoutRate=dropoutRate,
        l2_reg=l2_reg
    )

    # Define early stopping callback to prevent overfitting
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    if verbose: print("\nTraining the model...")
    history = model.fit(
        TrainingFeatures,
        trainingLabels,
        epochs=epochs,
        batch_size=batchSize,
        validation_split=trainingTestingSplit,
        verbose=verbose,
        class_weight=classWeights,
        callbacks=[earlyStopping]
    )

    # Evaluate the model on the test set
    results = model.evaluate(TestFeatures, testLabels, verbose=verbose)
    results_dict = dict(zip(model.metrics_names, results))
    Accuracy = results_dict.get('Accuracy', None)
    if verbose: 
        if Accuracy is not None:
            print(f'Accuracy: {Accuracy * 100:.2f}%')
        else:
            print('Accuracy metric is not available.')

    # Perform additional evaluation
    # evaluateModel(model, TestFeatures, testLabels)

    return model, history


def detectOverfitting(history, lossFunction):
    """Analyze the training history to detect potential overfitting.

    This function compares training and validation accuracy and loss to determine if the model is overfitting.

    Args:
        history (tf.keras.callbacks.History): The history object returned by the model's fit method.
        lossFunction (str): The loss function used during training.

    Returns:
        None

    Prints:
        - A warning message if overfitting is detected.
        - A confirmation message if the model is not overfitting.
    """
    print("----------------------------------")
    trainAcc = history.history['Accuracy']
    valAcc = history.history['val_Accuracy']
    trainLoss = history.history['loss']
    valLoss = history.history['val_loss']

    if lossFunction == 'squared_hinge':
        valLoss = np.array(valLoss) - 1

    acc_diff = trainAcc[-1] - valAcc[-1]
    loss_diff = valLoss[-1] - trainLoss[-1]

    if (acc_diff > 0.1) or (loss_diff > 0.1):
        print("Warning: The model is overfitting.")
    else:
        print("The model is not overfitting.")
    print("----------------------------------")


def findInputImportance(model, features, numSamples=100, shapSampleSize=200, plotSavePath=None):
    """Calculate and plot SHAP values to determine feature importance.

    Args:
        model (tf.keras.Model): The trained neural network model.
        features (pd.DataFrame): The feature columns of the dataset.
        num_samples (int): The number of samples to use for summarizing the background data.
        shap_sample_size (int): The number of samples to use for calculating SHAP values.
        plotSavePath (str): The directory path to save the plots.

    Returns:
        dict: A dictionary with features ordered from most to least important.

    Plots:
        - SHAP summary plot showing feature importance.
    """
    print("\nCalculating SHAP values for feature importance...")
    # Summarize the background data using shap.kmeans
    backgroundData = shap.kmeans(features, numSamples)

    # Create a SHAP explainer using KernelExplainer for more flexibility
    modelExplainer = shap.KernelExplainer(model.predict, backgroundData, seed=RANDOM_SEED)

    # Sample a subset of the dataset for SHAP value calculation
    shapSample = features.sample(shapSampleSize, random_state=RANDOM_SEED)

    # Calculate SHAP values for the sampled dataset
    shapValues = modelExplainer.shap_values(shapSample)

    # Ensure shapValues is 2D
    if isinstance(shapValues, list):
        shapValues = np.array(shapValues)
        if shapValues.ndim == 3:
            shapValues = shapValues[0]  # Assuming binary classification, take the first set of SHAP values

    # Reshape shapValues to 2D if necessary
    if shapValues.ndim == 3:
        shapValues = shapValues.reshape(shapValues.shape[0], -1)

    # Calculate mean absolute SHAP values for each feature
    meanAbsShapValues = pd.DataFrame(shapValues, columns=features.columns).abs().mean().sort_values(ascending=False)

    # Convert to dictionary
    featuresImportance = meanAbsShapValues.to_dict()

    # Plot the SHAP summary plot
    shap.summary_plot(shapValues, shapSample, show=False)
    if plotSavePath:
        plt.savefig(f"{plotSavePath}/shapSummaryPlot.png")

    plt.show()
    plt.clf()

    # Bar plot of feature importance
    shap.summary_plot(shapValues, shapSample, plot_type='bar', show=False)
    if plotSavePath:
        plt.savefig(f"{plotSavePath}/shapBarPlot.png")

    plt.show()
    plt.clf()

    return featuresImportance


def saveModel(model: tf.keras.Model, filePath:str):
    """Save the trained neural network model to the specified file path.

    The model is saved in Keras format. If the provided file path does not end with '.keras',
    the '.keras' extension is appended.

    Args:
        model (tf.keras.Model): The trained neural network model to be saved.
        filePath (str): The destination file path where the model will be saved.

    Returns:
        str: The file path where the model was saved.

    Raises:
        IOError: If the model cannot be saved to the specified path.
    """
    print(f"\nSaving model...")
    if not filePath.endswith('.keras'):
        filePath += '.keras'

    model.save(filePath)
    print(f"Model saved to '{filePath}'")
    return filePath


def plotLearningCurve(history, epochs, elapsedTime, lossFunction):
    """Plot the learning curves for accuracy and loss over epochs.

    This function creates an animated plot showing the progression of training and validation
    accuracy and loss over each epoch. The plot is styled with specified colors and saved as an animation.

    Args:
        history (tf.keras.callbacks.History): The history object returned by the model's fit method.
        epochs (int): Total number of epochs the model was trained for.
        elapsedTime (pd.Timedelta): The total time taken to train the model.
        lossFunction (str): The loss function used during training.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object representing the learning curves.
    """
    # Define color variables using hex codes
    backgroundColor = '#222222'
    lineColorTrain = '#FD4F59'
    lineColorVal = '#5BAFFC'
    textColor = '#DDDDDD'
    gridColor = '#5B5B5B'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # Adjusted height to 8
    fig.patch.set_facecolor(backgroundColor)
    ax1.set_facecolor(backgroundColor)
    ax2.set_facecolor(backgroundColor)

    # Determine the minimum and maximum values for loss and val_loss
    minLoss = min(min(history.history['loss']), min(history.history['val_loss']))
    maxLoss = max(max(history.history['loss']), max(history.history['val_loss']))

    # Adjust val_loss if the loss function is squared_hinge
    if lossFunction == 'squared_hinge':
        adjustedValLoss = np.array(history.history['val_loss']) - 1
    else:
        adjustedValLoss = np.array(history.history['val_loss'])

    def animate(i):
        ax1.clear()
        ax2.clear()

        if i < len(history.history['Accuracy']):
            ax1.plot(
                np.array(history.history['Accuracy'][:i]) * 100,
                color=lineColorTrain,
                label='Train Accuracy'
            )
            ax1.plot(
                np.array(history.history['val_Accuracy'][:i]) * 100,
                color=lineColorVal,
                label='Validation Accuracy'
            )
            ax2.plot(
                np.array(history.history['loss'][:i]),
                color=lineColorTrain,
                label='Train Loss'
            )
            ax2.plot(
                adjustedValLoss[:i],
                color=lineColorVal,
                label='Validation Loss'
            )
        else:
            ax1.plot(
                np.array(history.history['Accuracy']) * 100,
                color=lineColorTrain,
                label='Train Accuracy'
            )
            ax1.plot(
                np.array(history.history['val_Accuracy']) * 100,
                color=lineColorVal,
                label='Validation Accuracy'
            )
            ax2.plot(
                np.array(history.history['loss']),
                color=lineColorTrain,
                label='Train Loss'
            )
            ax2.plot(
                adjustedValLoss,
                color=lineColorVal,
                label='Validation Loss'
            )

        ax1.set_title(f'Model Accuracy (Elapsed Time: {elapsedTime})', color=textColor)
        ax1.set_ylabel('Accuracy (%)', color=textColor)
        ax1.set_xlabel('Epoch', color=textColor)
        ax1.legend(loc='upper left')
        ax1.set_ylim([0, 100])  # Fix the y-axis scale for Accuracy
        ax1.set_xlim([0, epochs])  # Fix the x-axis scale
        ax1.tick_params(axis='x', colors=textColor)
        ax1.tick_params(axis='y', colors=textColor)
        ax1.grid(True, color=gridColor, linestyle='--', linewidth=0.5)

        ax2.set_title('Model Loss', color=textColor)
        ax2.set_ylabel('Loss', color=textColor)
        ax2.set_xlabel('Epoch', color=textColor)
        ax2.legend(loc='upper left')

        ax2.set_ylim([0, 1])
        ax2.set_xlim([0, epochs])  # Fix the x-axis scale
        ax2.tick_params(axis='x', colors=textColor)
        ax2.tick_params(axis='y', colors=textColor)
        ax2.grid(True, color=gridColor, linestyle='--', linewidth=0.5)

    # Calculate the number of frames for the 5-second pause at the end
    pauseFrames = 5 * 30  # 5 seconds at 30 fps

    # Use the length of the history data for frames plus the pause frames
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(history.history['Accuracy']) + pauseFrames,
        interval=50,
        repeat=False
    )

    # Adjust layout to prevent overlap
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)

    plt.show()

    return ani


def evaluateModel(model, TestFeatures, actualLabels, threshold=0.3):
    """Evaluate the trained model's performance on the test dataset.

    This function calculates the predicted probabilities, converts them to binary labels based on a threshold,
    and prints the classification report and confusion matrix.

    Args:
        model (tf.keras.Model): The trained neural network model.
        TestFeatures (pd.DataFrame): The feature columns of the test dataset.
        actualLabels (pd.Series): The true labels of the test dataset.
        threshold (float, optional): The probability threshold for converting predicted probabilities to binary labels. Defaults to 0.3.

    Returns:
        None

    Prints:
        - Classification report showing precision, recall, f1-score, and support.
        - Confusion matrix showing true positives, true negatives, false positives, and false negatives.
    """
    predictedProb = model.predict(TestFeatures)
    predictedLabel = (predictedProb > threshold).astype("int32")
    print("Classification Report:")
    print(classification_report(actualLabels, predictedLabel, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(actualLabels, predictedLabel))


def loadModel(modelPath:str):
    """Load a trained neural network model from the specified file path.

    Args:
        modelPath (str): The path to the saved Keras model file.

    Returns:
        tf.keras.Model: The loaded neural network model.

    Raises:
        IOError: If the model cannot be loaded from the specified path.
    """
    print(f"\nLoading model from '{modelPath}'...")
    model = load_model(modelPath)
    print(f"Model loaded successfully")
    return model


def predictWithModel(model, inputData):
    """Generate predictions and intermediate layer outputs for given input data.

    This function creates an intermediate model that outputs the activations of all Dense layers,
    obtains the outputs for the provided input data, and returns both the final prediction and
    the outputs of each Dense layer.

    Args:
        model (tf.keras.Model): The trained neural network model.
        inputData (array-like): The input data for which predictions are to be made.

    Returns:
        tuple: A tuple containing:
            - prediction (array-like): The final output predictions of the model.
            - layer_outs (list of array-like): List of outputs from each Dense layer, including input data.

    Raises:
        ValueError: If the model does not contain any Dense layers.
    """
    # Include only input and Dense layers
    outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    if not outputs:
        raise ValueError("The model does not contain any Dense layers.")
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    # Get the outputs
    layer_outs = intermediate_model.predict(inputData)
    # Prepend the input data to layer_outs
    layer_outs = [inputData] + list(layer_outs)
    prediction = layer_outs[-1]
    return prediction, layer_outs


def predictWithModel2(model, features):
    """Predict target values using the trained model for the given features.

    Args:
        model (tf.keras.Model): The trained neural network model.
        features (pd.DataFrame or array-like): The feature data for which predictions are to be made.

    Returns:
        array-like: The predicted target values.

    Prints:
        - A message indicating that prediction is in progress.
    """
    print("\nPredicting target variable...")
    predictions = model.predict(features)
    return predictions


def runGridSearch(features, target, paramGrid: dict, CPULimitation: float = 0.7):
    """Perform a grid search to find the best combination of hyperparameters.

    This function tests various combinations of neural network architectures and training parameters
    to identify the configuration that yields the highest validation accuracy.

    Args:
        features (pd.DataFrame): The preprocessed feature columns.
        target (pd.Series): The target variable column.

    Returns:
        dict: A dictionary containing the best hyperparameters found during grid search.

    Prints:
        - The best accuracy achieved.
        - The corresponding best hyperparameters.
    """
    searchStartTime = pd.Timestamp.now()
    
    grid = ParameterGrid(paramGrid)
    totalGrid = len(grid)
    
    cpuCount = multiprocessing.cpu_count()
    nJobs = max(1, floor(cpuCount * CPULimitation))

    print(f"\n\nRunning grid search with {nJobs} CPU cores (power equivalent) for {totalGrid} parameters combinations...")

    manager = multiprocessing.Manager()
    bestAccuracy = manager.Value('d', 0.0)
    bestParams = manager.dict()
    progressWheelIndex = manager.Value('i', 0)
    totalElapsedTime = manager.Value('d', 0.0)
    valAccuracyHistory = manager.list()  # List to store validation accuracy history
    bestValAccuracyHistory = manager.list()  # List to store best validation accuracy history

    print(getProgressBar(0, progressWheelIndex.value) + "Best Accuracy: --.--%  |  Time remaining: --h--min --s  |  est. Finish Time: --h--", end='\r')

    def evaluate(params, idx, bestAccuracy, bestParams, valAccuracyHistory):
        trainStartTime = pd.Timestamp.now()

        optimizerName = params['optimizer']
        if optimizerName == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=params['learningRate'])
        elif optimizerName == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=params['learningRate'])
        elif optimizerName == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=params['learningRate'])
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        model, history = trainNeuralNet(
            features=features,
            target=target,
            layers=params['layers'],
            epochs=params['epochs'],
            batchSize=params['batchSize'],
            inputActivation=params['inputActivation'],
            hiddenActivation=params['hiddenActivation'],
            outputActivation=params['outputActivation'],
            metrics=params['metrics'],
            loss=params['loss'],
            optimizer=optimizer,
            dropoutRate=params['dropoutRate'],
            trainingTestingSplit=0.2,
            l2_reg=params['l2_reg'],
            verbose=0
        )        
        valAccuracy = history.history.get('val_Accuracy', [0])[-1]
        valAccuracyHistory.append(valAccuracy)  # Append validation accuracy to history
        if valAccuracy > bestAccuracy.value:
            bestAccuracy.value = valAccuracy
            bestParams.update(params)

        bestValAccuracyHistory.append(bestAccuracy.value)  # Append best validation accuracy to history

        # Update index for progress wheel
        progressWheelIndex.value += 1

        trainEndTime = pd.Timestamp.now()
        elapsedTrainTime = trainEndTime - trainStartTime
        totalElapsedTime.value += elapsedTrainTime.total_seconds()

        averageTime = (totalElapsedTime.value / progressWheelIndex.value) / nJobs

        estTimeRemaining = averageTime * (totalGrid - progressWheelIndex.value)
        estHours = int(estTimeRemaining // 3600)
        estMinutes = int((estTimeRemaining % 3600) // 60)
        estSeconds = int(estTimeRemaining % 60)

        estFinishTime = pd.Timestamp.now() + pd.Timedelta(seconds=estTimeRemaining)

        completion = progressWheelIndex.value / totalGrid
        estTimeStr = f"{estHours:02}h{estMinutes:02}min {estSeconds:02}s"
        estFinishTimeStr = f"{estFinishTime.hour:02}h{estFinishTime.minute:02}"
        print(getProgressBar(completion, progressWheelIndex.value) + f"Best Accuracy: {bestAccuracy.value * 100:.2f}%  |  Time remaining: {estTimeStr}  |  est. Finish Time: {estFinishTimeStr}", end='\r')
        
        return valAccuracy, params

    results = Parallel(n_jobs=nJobs)(
        delayed(evaluate)(params, idx, bestAccuracy, bestParams, valAccuracyHistory) for idx, params in enumerate(grid, 1)
    )

    print(getProgressBar(1, progressWheelIndex.value) + f"Best Accuracy: {bestAccuracy.value * 100:.2f}%", end='\r')
    searchEndTime = pd.Timestamp.now()

    elapsedTime = searchEndTime - searchStartTime
    print(f"\n\nGrid search completed in {int(elapsedTime.components.hours):02}h {int(elapsedTime.components.minutes):02}min {int(elapsedTime.components.seconds):02}s (average training time: {totalElapsedTime.value / totalGrid:.2f}s)")

    print(f"\nBest Accuracy: {bestAccuracy.value * 100:.2f}%")
    print("Best Parameters:")
    paramstr = "\n".join([f"\t{k}: {v}" for k, v in bestParams.items()])
    print(paramstr)

    modelDirectory = f'Models/TrainedModel_{bestParams["layers"]}_{bestParams["epochs"]}_{bestParams["batchSize"]}_{bestParams["dropoutRate"]}_{bestParams["l2_reg"]}_{bestParams["inputActivation"]}_{bestParams["hiddenActivation"]}_{bestParams["outputActivation"]}_{bestParams["metrics"]}_{bestParams["loss"]}_{bestParams["optimizer"]}({bestParams["learningRate"]})_0.2/'
    backgroundColor = '#222222'

    if not os.path.exists(modelDirectory):
        os.makedirs(modelDirectory)

    # Plot the validation accuracy history
    plt.figure(figsize=(10, 6), facecolor=backgroundColor)
    plt.gca().set_facecolor(backgroundColor)
    plt.plot(valAccuracyHistory, linestyle='-', color='#FD4F59', label='Validation Accuracy')
    plt.plot(bestValAccuracyHistory, linestyle='-', color='#5BAFFC', label='Best Validation Accuracy')
    plt.gca().tick_params(axis='y', colors='white')
    plt.gca().tick_params(axis='x', colors='white')
    plt.title('Grid Search Validation Accuracy History', color='white')
    plt.xlabel('Iteration', color='white')
    plt.ylabel('Validation Accuracy', color='white')
    plt.legend()
    plt.grid(True)
    
    # Save the plot before showing it
    plt.savefig(f'{modelDirectory}ValidationAccuracyHistory.png')
    plt.show()

    return dict(bestParams), modelDirectory


def runModelTraining():
    """Execute the entire model training pipeline, including preprocessing, hyperparameter optimization,
    training, evaluation, and saving of the final model.

    This function performs the following steps:
    1. Checks for available GPU and configures TensorFlow accordingly.
    2. Loads and preprocesses the dataset.
    3. Defines the neural network architecture and training parameters.
    4. Performs grid search to find the best hyperparameters.
    5. Trains the final model using the best hyperparameters.
    6. Evaluates the model for overfitting.
    7. Saves the trained model and the learning curve plot.

    Returns:
        str: The file path where the trained model is saved.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        Exception: If any step in the training pipeline fails.
    """
    print("TensorFlow version:", tf.__version__)
    print("NumPy version:", np.__version__)
    print("Random seed set to:", RANDOM_SEED)
    print()
    
    # Check if TensorFlow is using GPU
    physicalDevices = tf.config.list_physical_devices('GPU')
    if physicalDevices:
        try:
            tf.config.experimental.set_memory_growth(physicalDevices[0], True)
            print(f"Training using GPU: {physicalDevices}")
        except RuntimeError as e:
            print(e)
    else:
        print(f"Training using CPU with {multiprocessing.cpu_count()} cores")

    print()

    tableToDrop =[] # ['AverageHoursWorked']

    # Load and preprocess the dataset
    datasetFilePat = r'GeneratedDataSet\ModelDataSet.csv'
    features, target = loadAndPreprocessData(datasetFilePat, tableToDrop)

    # Define hyperparameter grid for grid search
    hyperparameterGrid = {
        'layers': [
            # [features.shape[1], 128, 64, 32, 1],
            [features.shape[1], 256, 128, 64, 1],
        ],
        'epochs': [100],
        'batchSize': [32],
        'dropoutRate': [0.3],
        'l2_reg': [0.001, 0.01],
        'learningRate': [0.0005, 0.001],
        "metrics": [
            ['Accuracy', 'Recall', 'Precision'],
            ['Accuracy', 'Precision'],
            ['Accuracy', 'Recall'],
            ['Accuracy'],
            ],
        'inputActivation': ['relu', 'tanh'],
        'hiddenActivation': ['relu', 'tanh'],
        'outputActivation': ['sigmoid'],
        'loss': ['binary_crossentropy', 'mean_squared_error'],
        'optimizer': ['adam']
    }

    CPULimitation = 1.0

    # Start hyperparameter grid search
    print("\nStarting Grid Search for Hyperparameter Optimization...")
    bestParams, modelDirectory = runGridSearch(features, target, hyperparameterGrid, CPULimitation)

    # Create the model directory after finding the best parameters
    os.makedirs(os.path.dirname(modelDirectory), exist_ok=True)

    defaultMappingFilePath = r'GeneratedDataSet\MappingValues.csv'
    updatedMappingFilePath = modelDirectory + 'MappingValues.csv'
    saveCustomMapping(defaultMappingFilePath, updatedMappingFilePath, tableToDrop)

    # Record the start time of training
    startTrainingTime = pd.Timestamp.now()

    # Use bestParams to train the final model
    optimizer = tf.keras.optimizers.Adam(learning_rate=bestParams['learningRate'])
    model, history = trainNeuralNet(
        features=features,
        target=target,
        layers=bestParams['layers'],
        epochs=bestParams['epochs'],
        batchSize=bestParams['batchSize'],
        inputActivation=bestParams['inputActivation'],
        hiddenActivation=bestParams['hiddenActivation'],
        outputActivation=bestParams['outputActivation'],
        metrics=bestParams['metrics'],
        loss=bestParams['loss'],
        optimizer=optimizer,
        dropoutRate=bestParams['dropoutRate'],
        trainingTestingSplit=0.2,
        l2_reg=bestParams['l2_reg'],
        verbose=1
    )

    # Record the end time of training
    endTrainingTime = pd.Timestamp.now()
    elapsedTime = endTrainingTime - startTrainingTime

    print(f"Training time: {elapsedTime}")

    # Check for overfitting
    detectOverfitting(history, bestParams['loss'])

    trainAccuracy = history.history['Accuracy'][-1]
    validationAccuracy = history.history['val_Accuracy'][-1]

    modelName = modelDirectory + f"Model_{trainAccuracy:.2f}_{validationAccuracy:.2f}_{elapsedTime.total_seconds()}s"

    # Save the trained model
    savePath = saveModel(model, modelName)

    # Save the model parameters to a file
    paramsFilePath = modelName + ".params"
    with open(paramsFilePath, 'w') as paramsFile:
        json.dump(bestParams, paramsFile)
    print(f"Model parameters saved as '{paramsFilePath}'")

    # Plot and save the learning curve
    plot = plotLearningCurve(history, bestParams['epochs'], elapsedTime, bestParams['loss'])

    # Save the plot as a gif using PillowWriter
    print(f"Saving learning curve gif...")
    plot.save(f'{modelName}.gif', writer=animation.PillowWriter(fps=30))
    print(f"Learning curve saved as '{modelName}.gif'")


    # Find feature importance using SHAP values 
    featuresImportance = findInputImportance(model, features, plotSavePath=modelDirectory)

    # Save the feature importance to a file
    importanceFilePath = modelDirectory + 'FeatureImportance.csv'
    
    featuresImportance = pd.DataFrame(featuresImportance.items(), columns=['Feature', 'Importance'])
    featuresImportance.to_csv(importanceFilePath, index=False)
    print(f"Feature importance saved as '{importanceFilePath}'")


    # Return the path where the model was saved
    return savePath


if __name__ == '__main__':
    modelPath = runModelTraining()