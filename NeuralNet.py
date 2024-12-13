import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

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

def buildNeuralNetModel(layers:list[int], inputActivation:str, hiddenActivation:str, outputActivation:str, metrics:list, loss:str, optimizer:str, dropoutRate:float=0.5, l2_reg:float=0.01):
    """
    Build a neural network model with the given layers
    """
    model = Sequential()
    
    # Add input layer with Input layer
    model.add(tf.keras.layers.Input(shape=(layers[0],)))
    model.add(Dense(layers[1], activation=inputActivation, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropoutRate))
    
    # Add hidden layers with dropout and L2 regularization
    for neurons in layers[2:-1]:
        model.add(Dense(neurons, activation=hiddenActivation, kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropoutRate))
        
    # Add output layer
    model.add(Dense(layers[-1], activation=outputActivation, kernel_regularizer=l2(l2_reg)))
    
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

def trainNeuralNet(features, target, layers:list[int], epochs:int, batchSize:int, inputActivation:str='relu', hiddenActivation:str='relu', outputActivation:str='sigmoid', metrics:list=['Accuracy'], loss:str='binary_crossentropy',  optimizer:str='adam', dropoutRate:float=0.5, trainingTestingSplit:float=0.2, l2_reg:float=0.01):
    """
    Train a neural network model on the given dataset
    """
    # Split the data into training and testing sets
    TrainingFeatures, TestFeatures, trainingLabels, testLabels = train_test_split(features, target, test_size=trainingTestingSplit, random_state=42, stratify=target)

    # Reset index of training labels
    trainingLabels = trainingLabels.reset_index(drop=True)

    # Compute class weights using sklearn
    classes = np.unique(trainingLabels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=trainingLabels)
    classWeights = {cls: weight for cls, weight in zip(classes, class_weights)}
    print(f"Computed class weights: {classWeights}")

    # Build the model
    model = buildNeuralNetModel(layers=layers, inputActivation=inputActivation, hiddenActivation=hiddenActivation, outputActivation=outputActivation, metrics=metrics, loss=loss, optimizer=optimizer, dropoutRate=dropoutRate, l2_reg=l2_reg)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    print("Training the model...")
    history = model.fit(
        TrainingFeatures,
        trainingLabels,
        epochs=epochs,
        batch_size=batchSize,
        validation_split=trainingTestingSplit,
        verbose=1,
        class_weight=classWeights,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    results = model.evaluate(TestFeatures, testLabels)
    results_dict = dict(zip(model.metrics_names, results))
    Accuracy = results_dict.get('Accuracy', None)
    if Accuracy is not None:
        print(f'Accuracy: {Accuracy * 100:.2f}%')
    else:
        print('Accuracy metric is not available.')

    # Additional evaluation
    evaluateModel(model, TestFeatures, testLabels)

    return model, history

def detectOverfitting(history, lossFunction):
    print("----------------------------------")
    trainAcc = history.history['Accuracy']
    valAcc = history.history['val_Accuracy']
    trainLoss = history.history['loss']
    valLoss = history.history['val_loss']

    if lossFunction == 'squared_hinge':
        valLoss = np.array(valLoss) - 1

    if (trainAcc[-1] - valAcc[-1] > 0.1) or (valLoss[-1] - trainLoss[-1] > 0.1):
        print("Warning: The model is overfitting.")
    else:
        print("The model is not overfitting.")
    print("----------------------------------")

def saveModel(model:tf.keras.Model, filePath:str):
    print(f"\nSaving model...")
    if not filePath.endswith('.keras'):
        filePath += '.keras'

    model.save(filePath)
    print(f"Model saved to '{filePath}'")
    return filePath

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

        if i < len(history.history['Accuracy']):
            ax1.plot(np.array(history.history['Accuracy'][:i]) * 100, color=lineColorTrain, label='Train Accuracy')
            ax1.plot(np.array(history.history['val_Accuracy'][:i]) * 100, color=lineColorVal, label='Validation Accuracy')
            ax2.plot(np.array(history.history['loss'][:i]), color=lineColorTrain, label='Train Loss')
            ax2.plot(adjustedValLoss[:i], color=lineColorVal, label='Validation Loss')
        else:
            ax1.plot(np.array(history.history['Accuracy']) * 100, color=lineColorTrain, label='Train Accuracy')
            ax1.plot(np.array(history.history['val_Accuracy']) * 100, color=lineColorVal, label='Validation Accuracy')
            ax2.plot(np.array(history.history['loss']), color=lineColorTrain, label='Train Loss')
            ax2.plot(adjustedValLoss, color=lineColorVal, label='Validation Loss')

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
    ani = animation.FuncAnimation(fig, animate, frames=len(history.history['Accuracy']) + pauseFrames, interval=50, repeat=False)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)

    plt.show()

    return ani

def evaluateModel(model, TestFeatures, actualLabels, threshold=0.3):
    predictedProb = model.predict(TestFeatures)
    predictedLabel = (predictedProb > threshold).astype("int32")
    print("Classification Report:")
    print(classification_report(actualLabels, predictedLabel, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(actualLabels, predictedLabel))


def loadModel(modelPath:str):
    """
    Load a trained model from the given file path
    """
    print(f"\nLoading model from '{modelPath}'...")
    model = load_model(modelPath)
    print(f"Model loaded successfully")
    return model

def predictWithModel(model, inputData):
    # Include only input and Dense layers
    outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    # Get the outputs
    layer_outs = intermediate_model.predict(inputData)
    # Prepend the input data to layer_outs
    layer_outs = [inputData] + list(layer_outs)
    prediction = layer_outs[-1]
    return prediction, layer_outs


def predict_with_model(model, features):
    """
    return the predicted value
    """
    print("\nPredicting target variable...")
    predictions = model.predict(features)
    return predictions

def testModel(modelPath):
    print("Testing the model\n")
    model = loadModel(modelPath)

    print(f"\nLoding and preprocessing data...")
    features, target = loadAndPreprocessData(r'D:/Cesi/Ripo/Cesi/FISE3/4_AI/AI-Project/GeneratedDataSet/ModelDataSet.csv')

    # Select one sample where Attrition=1
    sample_one = features[target == 1].iloc[0]
    # Select one sample where Attrition=0
    sample_zero = features[target == 0].iloc[0]

    # Prepare the samples for prediction
    sample_one_array = sample_one.values.reshape(1, -1)
    sample_zero_array = sample_zero.values.reshape(1, -1)

    print("Predicting with the model...")
    # Make predictions
    print("with data:", sample_one_array)
    prediction_one = predict_with_model(model, sample_one_array)
    print()
    print("with data:", sample_zero_array)
    prediction_zero = predict_with_model(model, sample_zero_array)

    # Print predictions
    print(f"Predicted Attrition: {prediction_one[0][0]:.2f} (Expected: near 1)")
    print(f"Predicted Attrition: {prediction_zero[0][0]:.2f} (Expected: near 0)")



def runModelTraining():
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
    hiddenLayers = [32, 16, 8]
    outputLayer = 1  # Binary classification

    epochs = 100  # Reduced number of epochs to prevent overfitting
    batchSize = 20  # Standard batch size
    dropoutRate = 0.3  # Increased dropout rate to prevent overfitting
    l2Reg = 0.005  # L2 regularization factor

    # all activation functions:
    # https://keras.io/api/layers/activations/

    inputActivation = 'tanh'
    hiddenActivation = 'tanh'
    outputActivation = 'sigmoid'

    metrics = ['precision', 'Accuracy']

    # all loss functions:
    # https://keras.io/api/losses/

    loss = 'binary_crossentropy'

    # all optimizers:
    # https://keras.io/api/optimizers/
    learningRate = 0.0005
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

    modelDirectory = f'Models\\TrainedModel_{layers}_{epochs}_{batchSize}_{dropoutRate}_{l2Reg}_{inputActivation}_{hiddenActivation}_{outputActivation}_{metrics}_{loss}_{optimizerName}({learningRate})_{trainingTestingSplit}\\'
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(modelDirectory), exist_ok=True)

    startTrainingTime = pd.Timestamp.now()
    model, history = trainNeuralNet(features=features, target=target, layers=layers, epochs=epochs, batchSize=batchSize, inputActivation=inputActivation, hiddenActivation=hiddenActivation, outputActivation=outputActivation, metrics=metrics, loss=loss, optimizer=optimizer, dropoutRate=dropoutRate, trainingTestingSplit=trainingTestingSplit, l2_reg=l2Reg)
    endTrainingTime = pd.Timestamp.now()
    elapsedTime = endTrainingTime - startTrainingTime

    print(f"Training time: {elapsedTime}")

    detectOverfitting(history, loss)

    trainAccuracy = history.history['Accuracy'][-1]
    validationAccuracy = history.history['val_Accuracy'][-1]

    modelName = modelDirectory + f"Model_{trainAccuracy:.2f}_{validationAccuracy:.2f}_{elapsedTime.total_seconds()}s"

    savePath = saveModel(model, modelName)

    plot = plotLearningCurve(history, epochs, elapsedTime, loss)

    # Save the plot as a gif using PillowWriter
    print(f"Saving learning curve gif'...")
    plot.save(f'{modelName}.gif', writer=animation.PillowWriter(fps=30))
    print(f"Learning curve saved as '{modelName}.gif'")

    # return model path
    return savePath


if __name__ == '__main__':
    modelPath = runModelTraining()