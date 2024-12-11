import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def loadAndPreprocessData(filePath):
    # Load the dataset
    data = pd.read_csv(filePath)

    # Strip any leading or trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Drop EmployeeID as it is not a feature
    data = data.drop('EmployeeID', axis=1)

    # Encode categorical variables
    labelEncoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        labelEncoders[column] = LabelEncoder()
        data[column] = labelEncoders[column].fit_transform(data[column])

    # Separate features and target variable
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']

    # Normalize all features to the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    return X, y

def buildNeuralNetModel(layers):
    model = Sequential()
    # Add input layer
    model.add(Dense(layers[0], input_dim=layers[0], activation='relu'))
    # Add hidden layers
    for neurons in layers[1:-1]:
        model.add(Dense(neurons, activation='relu'))
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))  # Change to 1 neuron for binary classification

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def trainNeuralNet(X, y, layers, epochs=50, batchSize=10):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = buildNeuralNetModel(layers)

    # Train the model
    print("Training the model...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, validation_split=0.2, verbose=1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Save the model
    model.save('trained_model.h5')
    print("Model saved to 'trained_model.h5'")

    return model, history


def saveModel(model, filePath):
    if not filePath.endswith('.h5'):
        filePath += '.h5'

    model.save(filePath)
    print(f"Model saved to '{filePath}'")




def loadAndUseModel(filePath, layers):
    # Load the model
    model = load_model('trained_model.h5')
    print("Model loaded from 'trained_model.h5'")

    # Load and preprocess the data
    X, y = loadAndPreprocessData(filePath)

    # Evaluate the model
    loss, accuracy = model.evaluate(X, y)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    return model


def predictWithModel(model, data):
    # Make predictions
    predictions = model.predict(data)
    predictions = [1 if x > 0.5 else 0 for x in predictions]
    return predictions


def predictDataWithModel(model, data):
    # Load the dataset
    data = pd.read_csv(data)

    # Strip any leading or trailing spaces from column names
    data.columns = data.columns.str.strip()

    # Drop EmployeeID as it is not a feature
    if 'EmployeeID' in data.columns:
        data = data.drop('EmployeeID', axis=1)

    # Encode categorical variables
    labelEncoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        labelEncoders[column] = LabelEncoder()
        data[column] = labelEncoders[column].fit_transform(data[column])

    # Normalize all features to the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)

    # Make predictions
    predictions = predictWithModel(model, data)
    return predictions



def plotLearningCurve(history, epoch):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # set the x-axis limit from 0 to the number of epochs
    ax1.set_xlim(0, epoch)
    ax2.set_xlim(0, epoch)

    def animate(i):
        ax1.clear()
        ax2.clear()

        ax1.plot(history.history['accuracy'][:i], color='#FD4F59', label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'][:i], color='#5BAFFC', label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='upper left')
        ax1.set_ylim([0, 1])  # Fix the y-axis scale for accuracy

        ax2.plot(history.history['loss'][:i], color='#FD4F59', label='Train Loss')
        ax2.plot(history.history['val_loss'][:i], color='#5BAFFC', label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper left')
        ax2.set_ylim([0, 1])  # Fix the y-axis scale for accuracy


    ani = animation.FuncAnimation(fig, animate, frames=len(history.history['accuracy']), repeat=False)
    plt.show()



if __name__ == "__main__":
    # Check if TensorFlow is using GPU
    print("")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Using GPU: {physical_devices}")
    else:
        print("Using CPU")

    filePath = r'GeneratedDataSet\ModelDataSet.csv'
    X, y = loadAndPreprocessData(filePath)

    inputLayer = X.shape[1]
    hiddenLayers = [32, 32, 16]
    outputLayer = 1  # Change to 1 for binary classification

    epochs = 50 # epochs is the number of times the model will be trained on the dataset
    batchSize = 100 # batch_size is the number of samples that will be used in each training iteration

    print(f"""
Generating a neural network with:
- Input layer: {inputLayer} neurons
- Hidden layers: {hiddenLayers}
- Output layer: {outputLayer} neuron{'s' if outputLayer > 1 else ''}

Training the model with:
- Epochs: {epochs}
- Batch size: {batchSize}
""")

    layers = [inputLayer] + hiddenLayers + [outputLayer]
    
    model, history = trainNeuralNet(X=X, y=y, layers=layers, epochs=epochs, batchSize=batchSize)
    saveModel(model, f'Models\TrainAttritionModel{layers}')

    plotLearningCurve(history, epochs)
    