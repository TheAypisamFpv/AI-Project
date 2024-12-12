import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def loadAndPreprocessData(filePath: str):
    """
    Load the dataset from the given file path and preprocess it
    """
    # Load the dataset
    data = pd.read_csv(filePath)

    # Strip any leading or trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Drop EmployeeID as it is not a feature
    data = data.drop('EmployeeID', axis=1)

    # Handle missing values
    data = data.dropna()

    # Encode categorical variables
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

def buildNeuralNetModel(layers:list[int], inputActivation:str, hiddenActivation:str, outputActivation:str, loss:str, optimizer:str, dropoutRate:float=0.5, l2_reg:float=0.01):
    """
    Build a neural network model with the given layers
    """

    model = Sequential()
    
    # Add input layer with L2 regularization
    model.add(Dense(layers[0], input_dim=layers[0], activation=inputActivation, kernel_regularizer=l2(l2_reg)))
    
    # Add hidden layers with dropout and L2 regularization
    for neurons in layers[1:-1]:
        model.add(Dense(neurons, activation=hiddenActivation, kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropoutRate))
        
    # Add output layer
    model.add(Dense(1, activation=outputActivation, kernel_regularizer=l2(l2_reg)))  # Change to 1 neuron for binary classification

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def trainNeuralNet(features, target, layers:list[int], epochs:int, batchSize:int, inputActivation:str='relu', hiddenActivation:str='relu', outputActivation:str='sigmoid', loss:str='binary_crossentropy',  optimizer:str='adam', dropoutRate:float=0.5, trainingTestingSplit:float=0.2, l2_reg:float=0.01):
    """
    Train a neural network model on the given dataset
    """
    # Split the data into training and testing sets
    TrainingFeatures, TestFeatures, trainingLabels, testLabels = train_test_split(features, target, test_size=trainingTestingSplit)

    # Reset index of training labels
    trainingLabels = trainingLabels.reset_index(drop=True)

    # Check for class imbalance and calculate class weights
    classWeights = None
    if len(np.unique(trainingLabels)) > 1:
        classWeights = {0: (1 / np.bincount(trainingLabels)[0]), 1: (1 / np.bincount(trainingLabels)[1])}
        print(f"Class weights: {classWeights}")

    # Build the model
    model = buildNeuralNetModel(layers=layers, inputActivation=inputActivation, hiddenActivation=hiddenActivation, outputActivation=outputActivation, loss=loss, optimizer=optimizer, dropoutRate=dropoutRate, l2_reg=l2_reg)

    # Early stopping callback
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    print("Training the model...")
    history = model.fit(TrainingFeatures, trainingLabels, epochs=epochs, batch_size=batchSize, validation_split=0.2, verbose=1, class_weight=classWeights)

    # Evaluate the model
    loss, accuracy = model.evaluate(TestFeatures, testLabels)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Additional evaluation
    evaluateModel(model, TestFeatures, testLabels)

    return model, history

def detectOverfitting(history, lossFunction):
    print("----------------------------------")
    trainAcc = history.history['accuracy']
    valAcc = history.history['val_accuracy']
    trainLoss = history.history['loss']
    valLoss = history.history['val_loss']

    if lossFunction == 'squared_hinge':
        valLoss = np.array(valLoss) - 1

    if (trainAcc[-1] - valAcc[-1] > 0.1) or (valLoss[-1] - trainLoss[-1] > 0.1):
        print("Warning: The model is overfitting.")
    else:
        print("The model is not overfitting.")
    print("----------------------------------")

def saveModel(model, filePath):
    print(f"\nSaving model...")
    if not filePath.endswith('.keras'):
        filePath += '.keras'

    model.save(filePath)
    print(f"Model saved to '{filePath}'")

def plotLearningCurve(history, epochs, elapsedTime, lossFunction):
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

        if i < len(history.history['accuracy']):
            ax1.plot(np.array(history.history['accuracy'][:i]) * 100, color=lineColorTrain, label='Train Accuracy')
            ax1.plot(np.array(history.history['val_accuracy'][:i]) * 100, color=lineColorVal, label='Validation Accuracy')
            ax2.plot(np.array(history.history['loss'][:i]), color=lineColorTrain, label='Train Loss')
            ax2.plot(adjustedValLoss[:i], color=lineColorVal, label='Validation Loss')
        else:
            ax1.plot(np.array(history.history['accuracy']) * 100, color=lineColorTrain, label='Train Accuracy')
            ax1.plot(np.array(history.history['val_accuracy']) * 100, color=lineColorVal, label='Validation Accuracy')
            ax2.plot(np.array(history.history['loss']), color=lineColorTrain, label='Train Loss')
            ax2.plot(adjustedValLoss, color=lineColorVal, label='Validation Loss')

        ax1.set_title(f'Model Accuracy (Elapsed Time: {elapsedTime})', color=textColor)
        ax1.set_ylabel('Accuracy (%)', color=textColor)
        ax1.set_xlabel('Epoch', color=textColor)
        ax1.legend(loc='upper left')
        ax1.set_ylim([0, 100])  # Fix the y-axis scale for accuracy
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
    ani = animation.FuncAnimation(fig, animate, frames=len(history.history['accuracy']) + pauseFrames, interval=50, repeat=False)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)

    plt.show()

    return ani

def evaluateModel(model, TestFeatures, actualLabels):
    predictedLabel = (model.predict(TestFeatures) > 0.5).astype("int32")
    print("Classification Report:")
    print(classification_report(actualLabels, predictedLabel, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(actualLabels, predictedLabel))


def main():
    # Check if TensorFlow is using GPU
    print("")
    physicalDevices = tf.config.list_physical_devices('GPU')
    if physicalDevices:
        try:
            tf.config.experimental.set_memory_growth(physicalDevices[0], True)
            print(f"Using GPU: {physicalDevices}")
        except RuntimeError as e:
            print(e)
    else:
        print("Using CPU")

    filePath = r'GeneratedDataSet\ModelDataSet.csv'
    features, target = loadAndPreprocessData(filePath)

    trainingTestingSplit = 0.3  # % of the data will be used for testing

    inputLayer = features.shape[1]  # 24 input features
    hiddenLayers = [16, 8]
    outputLayer = 1  # Binary classification

    epochs = 100  # Reduced number of epochs to prevent overfitting
    batchSize = 25  # Standard batch size
    dropoutRate = 0.6  # Increased dropout rate to prevent overfitting
    l2Reg = 0.05  # L2 regularization factor

    # all activation functions:
    # https://keras.io/api/layers/activations/

    inputActivation = 'tanh'
    hiddenActivation = 'tanh'
    outputActivation = 'sigmoid'

    # all loss functions:
    # https://keras.io/api/losses/

    loss = 'squared_hinge'

    # all optimizers:
    # https://keras.io/api/optimizers/
    learningRate = 0.005
    optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
    optimizerName = optimizer.__class__.__name__

    print(f"""
Generating a neural network with:
- Input layer: {inputLayer} neurons
- Hidden layers: {hiddenLayers}
- Output layer: {outputLayer} neuron{'s' if outputLayer > 1 else ''}

Training the model with:
- Epochs: {epochs}
- Batch size: {batchSize}
- Optimizer: {optimizerName} (learning rate: {learningRate})
- Training/testing split: {(1 - trainingTestingSplit) * 100}/{trainingTestingSplit * 100}%
- Dropout rate: {dropoutRate}
- L2 regularization: {l2Reg}
""")

    layers = [inputLayer] + hiddenLayers + [outputLayer]

    modelDirectory = f'Models\\TrainedModel_{layers}_{epochs}_{batchSize}_{dropoutRate}_{l2Reg}_{inputActivation}_{hiddenActivation}_{outputActivation}_{loss}_{optimizerName}({learningRate})_{trainingTestingSplit}\\'
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(modelDirectory), exist_ok=True)

    startTrainingTime = pd.Timestamp.now()
    model, history = trainNeuralNet(features=features, target=target, layers=layers, epochs=epochs, batchSize=batchSize, inputActivation=inputActivation, hiddenActivation=hiddenActivation, outputActivation=outputActivation, loss=loss, optimizer=optimizer, dropoutRate=dropoutRate, trainingTestingSplit=trainingTestingSplit, l2_reg=l2Reg)
    endTrainingTime = pd.Timestamp.now()
    elapsedTime = endTrainingTime - startTrainingTime

    print(f"Training time: {elapsedTime}")

    detectOverfitting(history, loss)

    trainAccuracy = history.history['accuracy'][-1]
    validationAccuracy = history.history['val_accuracy'][-1]

    modelName = modelDirectory + f"Model_{trainAccuracy:.2f}_{validationAccuracy:.2f}_{elapsedTime.total_seconds()}s"

    saveModel(model, modelName)

    plot = plotLearningCurve(history, epochs, elapsedTime, loss)

    # Save the plot as a gif using PillowWriter
    print(f"Saving learning curve gif'...")
    plot.save(f'{modelName}.gif', writer=animation.PillowWriter(fps=30))
    print(f"Learning curve saved as '{modelName}.gif'")


if __name__ == '__main__':
    main()