import json
import time
from sklearn.cluster import AgglomerativeClustering
from NeuralNet import predictWithModel, loadModel, findInputImportance
from tkinter import Tk, filedialog
import tensorflow as tf
import pandas as pd
import numpy as np
import pyperclip
import threading
import difflib
import pygame
import csv
import os


# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class NeuralNetApp:
    def __init__(self):
        # Colors
        self.NEGATIVE_COLOR = pygame.Color("#FD4F59")
        self.POSITIVE_COLOR = pygame.Color("#5BAFFC")
        self.TEXT_COLOR = pygame.Color("#DDDDDD")
        self.BACKGROUND_COLOR = pygame.Color("#222222")
        self.INPUT_BACKGROUND_COLOR = pygame.Color("#333333")
        self.HOVER_COLOR = pygame.Color("#AAAAAA")
        self.ACTIVE_COLOR = pygame.Color("#555555")  # Changed ACTIVE_COLOR for better visibility
        
        # Font and input box settings
        self.FONT_SIZE = 15 # Font size for input text
        # Width and height of input boxes
        self.INPUT_BOX_TOP_MARGIN = 10
        self.INPUT_BOX_BOTTOM_MARGIN = 10
        self.INPUT_BOX_WIDTH = 140
        self.INPUT_BOX_HEIGHT = 20
        
        # Minimum window size
        self.MIN_WIDTH = 1000
        self.MIN_HEIGHT = 800
        self.TOP_BOTTOM_MARGIN = self.INPUT_BOX_TOP_MARGIN + self.INPUT_BOX_BOTTOM_MARGIN  # Margin at the top and bottom of the window
        self.MAX_HELP_TEXT_WIDTH = 400  # Maximum width for input help text
        self.INPUT_TEXT_HORIZONTAL_SPACING = 20  # Initial horizontal spacing
        self.INPUT_TEXT_VERTICAL_SPACING = 5    # Initial vertical spacing
        self.NORMALIZATION_RANGE = (-1, 1) # used to renormalize the input values to the range of the model

        # FPS setting
        self.fps = 60

        # Clustering settings
        self.clusterThreshold = 50  # Number of neurons with 
        self.enableClustering = True  # Variable to toggle clustering


        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Neural Network Prediction")
        self.screen = pygame.display.set_mode(
            (1400, 900), pygame.RESIZABLE
        )
        self.screen.fill(self.BACKGROUND_COLOR)
        self.font = pygame.font.Font(None, self.FONT_SIZE)
        self.cursor = pygame.Rect(0, -2, 2, self.INPUT_BOX_HEIGHT - 10)
        self.clock = pygame.time.Clock()

        # Initialize variables
        self.inputBoxes = []
        self.inputValues = {}
        self.cursorVisible = True
        self.cursorTimer = 0
        self.running = True
        self.prediction = None
        self.intermediateOutputs = None
        self.activeBox = None
        self.activeFeature = None
        self.modelFilePath = None
        self.model = None
        self.mapping = None
        self.featuresImportance = None
        self.lastInputValues = None
        self.lastInputChangeTime = time.time()

        # Margins for visualization
        self.leftMargin = 600
        self.rightMargin = 150
        self.topMargin = self.INPUT_BOX_TOP_MARGIN # top margin is the same as the start of the input box so that the input boxes are aligned with the visualization
        self.bottomMargin = self.INPUT_BOX_BOTTOM_MARGIN
        self.visualisationArea = pygame.Rect(
                self.leftMargin-10,
                self.topMargin-10,
                self.screen.get_width() - self.leftMargin - self.rightMargin + 20,
                self.screen.get_height() - self.topMargin - self.bottomMargin + 20
            )

        # Threading for predictions
        self.predictionThread = None
        self.predictionLock = threading.Lock()
        self.predictionReady = False

        # Store last valid prediction
        self.lastPrediction = None
        self.lastIntermediateOutputs = None

        # Variables for text selection
        self.selectedText = ""
        self.selectionStart = None
        self.selectionEnd = None

        # Connection width settings
        self.MIN_CONNECTION_WIDTH = 1
        self.MAX_CONNECTION_WIDTH = 10

    def isFloat(self, value):
        """Check if the string can be converted to a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def loadModelParams(self):
        """
        Load the mapping values for the model features from a CSV file.
        """

        directory = self.modelFilePath.rsplit("/", 1)[0] + "/"
        modelName = self.modelFilePath.rsplit("/", 1)[1].rsplit(".", 1)[0]
        if not directory:
            directory = "./"

        mappingFilePath = directory + r"MappingValues.csv"
        paramsFilePath = directory + modelName + '.params'
        print(f"\nLoading model parameters from {paramsFilePath}...")
        if not os.path.exists(paramsFilePath):
            print(f"No .params file found at {paramsFilePath}, using default random seed (42)")
        else:  
            # load the randomSeed in .params file (json format)
            with open(paramsFilePath, 'r') as file:
                params = json.load(file)
                if not 'randomSeed' in params:
                    print(f"No random seed found in {paramsFilePath}, using default seed (42)")
                try:
                    np.random.seed(int(params['randomSeed']))
                    tf.random.set_seed(int(params['randomSeed']))
                    print(f"Loaded random seed from {paramsFilePath}")
                except ValueError:
                    print(f"Invalid random seed in {paramsFilePath}, using default seed (42)")

                        
                        
        print("\nChecking for mapping values...")
        if not os.path.exists(mappingFilePath):
            # use default values
            errorMessage = "No mapping file found, using default values (this may not work for all models)"
            print("-" * len(errorMessage))
            print(errorMessage)
            print("-" * len(errorMessage))
            mappingFilePath = r"GeneratedDataSet\MappingValues.csv"
        else:
            print(f"Loading mapping values from {mappingFilePath}...")

        
        mapping = {}
        with open(mappingFilePath, mode='r') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            values = next(reader)
            for header, value in zip(headers, values):
                if value.startswith("[") and value.endswith("]"):
                    value = eval(value)
                mapping[header] = value

        self.mapping = mapping
        print(f"Loaded mapping values from {mappingFilePath}\n")

        # Load feature importance from CSV if it exists
        importanceFilePath = os.path.join(directory, "FeatureImportance.csv")
        print(f"Checking for feature importance file at {importanceFilePath}...")
        if os.path.exists(importanceFilePath):
            self.featuresImportance = pd.read_csv(importanceFilePath, index_col=0)["Importance"].to_dict()
            print(f"Loaded feature importance from {importanceFilePath}\n")
        else:
            self.featuresImportance = {feature: 1 for feature in self.mapping.keys()}
            print("Feature importance file not found, using default neuron size\n")
    

    def selectModelFile(self):
        """
        Open a file dialog to select a model file.
        """
        root = Tk()
        root.withdraw()  # Hide the root window
        filePath = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.keras"), ("All Files", "*.*")])
        root.destroy()
        return filePath


    def wrapText(self, text, font, maxWidth):
        """
        Wrap text to fit within a given width.
        """
        words = text.split(' ')
        lines = []
        currentLine = []
        currentWidth = 0

        for word in words:
            wordWidth, _ = font.size(word + ' ')
            if currentWidth + wordWidth <= maxWidth:
                currentLine.append(word)
                currentWidth += wordWidth
            else:
                lines.append(' '.join(currentLine))
                currentLine = [word]
                currentWidth = wordWidth

        if currentLine:
            lines.append(' '.join(currentLine))

        return lines


    def getPredictionThread(self):
        """
        Thread function to get a prediction from the model.
        """
        with self.predictionLock:
            if self.predictionReady:
                return
            
            self.predictionReady = True
        
        try:
            prediction, intermediateOutputs = self.getPrediction()
            with self.predictionLock:
                self.prediction = prediction
                self.intermediateOutputs = intermediateOutputs
                self.lastPrediction = prediction
                self.lastIntermediateOutputs = intermediateOutputs
                self.predictionReady = False
        except Exception as e:
            print(f"Error getting prediction: {e}")
            with self.predictionLock:
                self.predictionReady = False    

    def getPrediction(self):
        """
        Parse input values and get a prediction from the model.
        """
        inputData = []
        waitTime = 0.5
        print(time.time() - self.lastInputChangeTime)
        lastInputChangeTime = time.time() - self.lastInputChangeTime

        if self.lastInputValues != self.inputValues and self.lastInputValues:
            self.lastInputChangeTime = time.time()
            self.lastInputValues = self.inputValues.copy()
            return None, None

        if (lastInputChangeTime < waitTime or lastInputChangeTime > waitTime + 0.2) and self.lastInputValues:
            return None, None






        # # return nothing if the input values have changed within the waitTime
        # if lastInputChangeTime < waitTime and self.lastInputValues != self.inputValues:
        #     # self.lastInputChangeTime = time.time()
        #     # self.lastInputValues = self.inputValues.copy()
        #     return None, None
        
        # # # return None, None if the input values have not changed
        # if self.lastInputValues == self.inputValues:
        #     self.lastInputChangeTime = time.time()
        #     return None, None

        self.lastInputValues = self.inputValues.copy()
        # self.lastInputChangeTime = time.time()


        
        for inputName, inputValue in self.inputValues.items():
            if not inputValue:
                inputData.append(0)
                continue
            
            inputMapping = self.mapping[inputName]
            
            if isinstance(inputMapping[0], (int, float)):
                minValue, maxValue = inputMapping
                try:
                    inputValue = float(inputValue)
                except ValueError:
                    inputValue = 0
                normalizedValue = (float(inputValue) - minValue) / (maxValue - minValue) * (self.NORMALIZATION_RANGE[1] - self.NORMALIZATION_RANGE[0]) + self.NORMALIZATION_RANGE[0]
                normalizedValue = max(self.NORMALIZATION_RANGE[0], min(self.NORMALIZATION_RANGE[1], normalizedValue))
                inputData.append(normalizedValue)
                continue
            
            elif isinstance(inputMapping[0], str):
                inputMapping = [str(val).lower() for val in inputMapping]
                inputValue = inputValue.lower()
                closestMatch = difflib.get_close_matches(inputValue, inputMapping, n=1, cutoff=0.1)
                if closestMatch:
                    normalizedValue = (inputMapping.index(closestMatch[0]) / (len(inputMapping) - 1)) * (self.NORMALIZATION_RANGE[1] - self.NORMALIZATION_RANGE[0]) + self.NORMALIZATION_RANGE[0]
                    normalizedValue = max(self.NORMALIZATION_RANGE[0], min(self.NORMALIZATION_RANGE[1], normalizedValue))
                    inputData.append(normalizedValue)
                    continue
            else:
                raise ValueError(f"Invalid data type for feature mapping: {inputMapping} (got {type(inputMapping)}, expected list)")

            inputData.append(0)

        # Ensure inputData matches the expected input shape
        inputData = np.array([inputData])
        expectedInputShape = self.model.input_shape[1]
        if inputData.shape[1] != expectedInputShape:
            raise ValueError(f"Expected input shape ({expectedInputShape}) does not match provided input shape ({inputData.shape[1]})")

        prediction, intermediateOutputs = predictWithModel(self.model, inputData)
        return prediction, intermediateOutputs


    def interpolateColor(self, colorA, colorB, factor:float):
        """Interpolate between two colors with a given factor (0 to 1)."""

        if np.isnan(factor):
            return pygame.Color('#666666')

        factor = max(0, min(1, factor))
        
        return pygame.Color(
            int(colorA.r + (colorB.r - colorA.r) * factor),
            int(colorA.g + (colorB.g - colorA.g) * factor),
            int(colorA.b + (colorB.b - colorA.b) * factor)
        )

    def clusterNeurons(self, layerOutputs):
        """
        Cluster neurons in a layer using hierarchical clustering.
        """
        if not self.enableClustering:
            return [(i,) for i in range(layerOutputs.shape[1])]  # No clustering if disabled

        numNeurons = layerOutputs.shape[1]
        if numNeurons < self.clusterThreshold:
            return [(i,) for i in range(numNeurons)]  # No clustering needed

        clustering = AgglomerativeClustering(n_clusters=self.clusterThreshold)
        labels = clustering.fit_predict(layerOutputs.T)
        clusters = [[] for _ in range(self.clusterThreshold)]
        for neuronIndex, clusterIndex in enumerate(labels):
            clusters[clusterIndex].append(neuronIndex)
        return clusters

    def visualizeNeurons(self, screen, model, intermediateOutputs):
        """
        Create a visualization of the neural network with neurons displaying their output values,
        and outgoing connections colored based on the neuron's output value.
        """
        timeStart = time.time()
        
        # Combine input data with intermediate outputs
        allOutputs = intermediateOutputs
    
        numLayers = len(allOutputs)
        maxNeurons = max([layerOutput.shape[1] for layerOutput in allOutputs])
    
        padding = 10
        startWidth = self.leftMargin + padding
        availableWidth = self.screen.get_width() - self.leftMargin - self.rightMargin - padding*2
        layerSpacing = availableWidth / (numLayers - 1)
    
        startHeight = self.topMargin + 10
        availableHeight = self.screen.get_height() - self.topMargin - self.bottomMargin - padding*2
    
        neuronRadius = max(10, int(min(layerSpacing, availableHeight / maxNeurons) / 2))
    
        layers = [layerOutput.shape[1] for layerOutput in allOutputs]
    
        # Extract weights from the model
        weights = [layer.get_weights()[0] for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        # lineWidth = 3
    
        self.screen.set_clip(self.visualisationArea)

        # Get feature importance
        maxImportance = max(self.featuresImportance.values())
        minImportance = min(self.featuresImportance.values())
        importanceRange = maxImportance - minImportance 
    
        def normalizeImportance(importance):
            defaultNeuronSize = 7
            if importanceRange == 0:
                return neuronRadius
            
            return defaultNeuronSize + (importance - minImportance) / importanceRange * defaultNeuronSize
    
        # First pass: sort connections by weight
        connections = []
        for i in range(len(layers) - 1):
            currentLayerSize = layers[i]
            nextLayerSize = layers[i + 1]
            x = startWidth + i * layerSpacing
            nextX = startWidth + (i + 1) * layerSpacing
    
            currentNeuronSpacing = availableHeight / currentLayerSize
            nextNeuronSpacing = availableHeight / nextLayerSize
    
            currentTotalLayerHeight = (currentLayerSize - 1) * currentNeuronSpacing
            currentYStart = startHeight + (availableHeight - currentTotalLayerHeight) / 2
    
            nextTotalLayerHeight = (nextLayerSize - 1) * nextNeuronSpacing
            nextYStart = startHeight + (availableHeight - nextTotalLayerHeight) / 2
    
            # Cluster neurons in the current and next layers
            currentClusters = self.clusterNeurons(allOutputs[i])
            nextClusters = self.clusterNeurons(allOutputs[i + 1])
    
            # Normalize weights for the current layer
            currentWeights = weights[i]
            maxWeight = np.max(np.abs(currentWeights))
            minWeight = np.min(np.abs(currentWeights))
            weightRange = maxWeight - minWeight
    
            for currentCluster in currentClusters:
                currentClusterOutput = np.mean([allOutputs[i][0][j] for j in currentCluster])
                # normalize the output to [0, 1] using the NORMALIZATION_RANGE
                normalizedOutput = (currentClusterOutput - self.NORMALIZATION_RANGE[0]) / (self.NORMALIZATION_RANGE[1] - self.NORMALIZATION_RANGE[0])
                color = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, normalizedOutput)
                y = currentYStart + np.mean([j * currentNeuronSpacing for j in currentCluster])
    
                for nextCluster in nextClusters:
                    nextY = nextYStart + np.mean([j * nextNeuronSpacing for j in nextCluster])
                    weight = np.mean([currentWeights[j, k] for j in currentCluster for k in nextCluster])
                    
                    # Adjust alpha channel based on weight
                    alphaFactor = (np.abs(weight) - minWeight) / weightRange
                    blendedColor = self.interpolateColor(color, self.BACKGROUND_COLOR, 1 - alphaFactor)
                    
                    connections.append((weight, (x, y, nextX, nextY, blendedColor)))
    
        # Sort connections by weight
        connections.sort(key=lambda conn: abs(conn[0]))
    
        # Draw connections
        for _, (x, y, nextX, nextY, blendedColor) in connections:
            pygame.draw.aaline(screen, blendedColor, (int(x), int(y)), (int(nextX), int(nextY)), blend=1)

            # lineSpaceing = 3
            # for offset in range(-lineWidth // lineSpaceing, lineWidth // lineSpaceing + 1):
            #     if offset == -lineWidth // 3 or offset == lineWidth // 3:
            #         pygame.draw.aaline(screen, blendedColor, (int(x), int(y + offset)), (int(nextX), int(nextY + offset)), blend=1)
            #     else:
            #         pygame.draw.line(screen, blendedColor, (int(x), int(y + offset)), (int(nextX), int(nextY + offset)))

    
        # Second pass: Draw all neurons and their values
        for i in range(len(layers)):
            layerSize = layers[i]
            x = startWidth + i * layerSpacing
            neuronSpacing = availableHeight / layerSize
            totalLayerHeight = (layerSize - 1) * neuronSpacing
            yStart = startHeight + (availableHeight - totalLayerHeight) / 2
    
            # Cluster neurons in the current layer
            clusters = self.clusterNeurons(allOutputs[i])
    
            for cluster in clusters:
                clusterOutput = np.mean([allOutputs[i][0][j] for j in cluster])
                # normalize the output to [0, 1] using the NORMALIZATION_RANGE
                normalizedOutput = (clusterOutput - self.NORMALIZATION_RANGE[0]) / (self.NORMALIZATION_RANGE[1] - self.NORMALIZATION_RANGE[0])
                color = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, normalizedOutput)
                y = yStart + np.mean([j * neuronSpacing for j in cluster])
                
                # Adjust neuron radius based on importance for input layer
                if i == 0:
                    feature = list(self.mapping.keys())[cluster[0]]
                    importance = self.featuresImportance.get(feature, 1)
                    adjustedRadius = int(normalizeImportance(importance))
                else:
                    adjustedRadius = neuronRadius
    
                pygame.draw.circle(screen, color, (int(x), int(y)), adjustedRadius)
    
                # Render neuron output value
                valueText = f"{clusterOutput:.2f}"
                textSurface = self.font.render(valueText, True, self.TEXT_COLOR)
                textRect = textSurface.get_rect(center=(int(x), int(y)))
                screen.blit(textSurface, textRect)
    
        # Reset clipping area
        self.screen.set_clip(None)
    
        print(f"Visualization took {time.time() - timeStart:.2f} seconds")

    def clearVisualizationArea(self):
        """
        Clear the visualization area.
        """
        self.screen.set_clip(self.visualisationArea)
        self.screen.fill(self.BACKGROUND_COLOR)
        self.screen.set_clip(None)

    def updateInputBoxes(self):
        """
        Update the input boxes based on the current window size.
        Creates them if they don't exist yet.
        """
        
        verticalSpacing = (self.screen.get_height() - self.INPUT_BOX_TOP_MARGIN - self.INPUT_BOX_BOTTOM_MARGIN - self.INPUT_BOX_HEIGHT) / len(self.mapping)
        inputBoxStartY = self.INPUT_BOX_TOP_MARGIN + self.INPUT_BOX_HEIGHT
        
        if not self.inputBoxes:
            for i, (feature, values) in enumerate(self.mapping.items()):
                inputBox = pygame.Rect(
                    10,
                    inputBoxStartY + int(verticalSpacing * i),
                    self.INPUT_BOX_WIDTH,
                    self.INPUT_BOX_HEIGHT
                )
                self.inputBoxes.append((feature, inputBox, values))
                self.inputValues[feature] = ""
        else:
            for i, (feature, inputBox, values) in enumerate(self.inputBoxes):
                newY = inputBoxStartY + int(verticalSpacing * i)
                self.inputBoxes[i] = (feature, pygame.Rect(10, newY, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT), values)
                

    def mainLoop(self, visualisation=False):
        self.screen.fill(self.BACKGROUND_COLOR)

        if not self.enableClustering:
            print("\n=============================================")
            print("Clustering is disabled. This may lead to a large number of neurons being visualized, thus slowing down the visualization.")
            print("Use clustering for models with a large number of neurons to improve performance.")
            print("=============================================\n")
        
        while self.running:
            # Set the clipping area for the visualization
            
            # top part
            self.screen.set_clip(
                pygame.Rect(
                    0,
                    0,
                    self.screen.get_width(),
                    self.visualisationArea.top
                )
            )
            self.screen.fill(self.BACKGROUND_COLOR)
    
            # bottom part
            self.screen.set_clip(
                pygame.Rect(
                    0,
                    self.visualisationArea.bottom,
                    self.screen.get_width(),
                    self.screen.get_height() - self.visualisationArea.bottom
                )
            )
            self.screen.fill(self.BACKGROUND_COLOR)
    
            # left part
            self.screen.set_clip(
                pygame.Rect(
                    0,
                    0,
                    self.visualisationArea.left,
                    self.screen.get_height()
                )
            )
            self.screen.fill(self.BACKGROUND_COLOR)
    
            # right part
            self.screen.set_clip(
                pygame.Rect(
                    self.visualisationArea.right,
                    0,
                    self.screen.get_width() - self.visualisationArea.right,
                    self.screen.get_height()
                )
            )
            self.screen.fill(self.BACKGROUND_COLOR)
    
            # Reset clipping area
            self.screen.set_clip(None)
    
            self.cursorTimer += 1
    
            # Toggle cursor visibility (if the window is focused)
            windowFocused = pygame.key.get_focused()
            if self.cursorTimer % (self.fps // 2) == 0 and windowFocused:
                self.cursorVisible = not self.cursorVisible
            elif not windowFocused:
                self.cursorVisible = False

    
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    newWidth = max(event.w, self.MIN_WIDTH)
                    newHeight = max(event.h, self.MIN_HEIGHT)
                    self.screen = pygame.display.set_mode(
                        (newWidth, newHeight), pygame.RESIZABLE
                    )
                    # Update input text spacing based on new window height
                    self.INPUT_TEXT_HORIZONTAL_SPACING = newHeight * 0.02  # 2% of window height
                    self.INPUT_TEXT_VERTICAL_SPACING = newHeight * 0.005    # 0.5% of window height

                    self.visualisationArea = pygame.Rect(
                        self.leftMargin-10,
                        self.topMargin-10,
                        self.screen.get_width() - self.leftMargin - self.rightMargin + 20,
                        self.screen.get_height() - self.topMargin - self.bottomMargin + 20
                    )

                    self.clearVisualizationArea()
                    self.lastInputValues = None
                    
                    # Update input box vertical positions based on new window height
                    if self.modelFilePath:
                        self.updateInputBoxes()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle mouse clicks
                    if self.modelFilePath is None:
                        # If no model is selected, check if "Select Model" button is clicked
                        if hasattr(self, 'selectModelButton') and self.selectModelButton.collidepoint(event.pos):
                            self.modelFilePath = self.selectModelFile()
                            if self.modelFilePath:
                                self.model = loadModel(self.modelFilePath)
                                self.loadModelParams()
                                self.updateInputBoxes()
                    else:
                        # If model is loaded, check if any input box is clicked
                        for feature, inputBox, values in self.inputBoxes:
                            if inputBox.collidepoint(event.pos):
                                self.activeBox = inputBox
                                self.activeFeature = feature
                                self.cursorTimer = 0
                
                elif event.type == pygame.KEYDOWN:
                # Handle keyboard input for active input box
                    if self.activeBox:
                        if event.key == pygame.K_RETURN:
                            self.activeBox = None
                        elif event.key == pygame.K_BACKSPACE:
                            self.inputValues[self.activeFeature] = self.inputValues[self.activeFeature][:-1]
                        elif event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                            # Handle paste (Ctrl+V)
                            clipboardText = pyperclip.paste()
                            self.inputValues[self.activeFeature] += clipboardText
                        elif event.key == pygame.K_c and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                            # Handle copy (Ctrl+C)
                            pyperclip.copy(self.inputValues[self.activeFeature])
                        elif event.key == pygame.K_UP:
                            # Navigate to the previous input box
                            currentIndex = next((i for i, (feature, inputBox, values) in enumerate(self.inputBoxes) if inputBox == self.activeBox), None)
                            if currentIndex is not None and currentIndex > 0:
                                self.activeBox = self.inputBoxes[currentIndex - 1][1]
                                self.activeFeature = self.inputBoxes[currentIndex - 1][0]
                                self.cursorTimer = 0
                        elif event.key == pygame.K_DOWN:
                            # Navigate to the next input box
                            currentIndex = next((i for i, (feature, inputBox, values) in enumerate(self.inputBoxes) if inputBox == self.activeBox), None)
                            if currentIndex is not None and currentIndex < len(self.inputBoxes) - 1:
                                self.activeBox = self.inputBoxes[currentIndex + 1][1]
                                self.activeFeature = self.inputBoxes[currentIndex + 1][0]
                                self.cursorTimer = 0
                        else:
                            self.inputValues[self.activeFeature] += event.unicode
    
            if self.modelFilePath:
                # Draw input fields
                for feature, inputBox, values in self.inputBoxes:
                    # Determine background color based on active state
                    boxBackgroundColor = self.ACTIVE_COLOR if inputBox == self.activeBox and windowFocused else self.INPUT_BACKGROUND_COLOR
    
                    # Calculate help text
                    wrappedText = self.wrapText(f"{feature}: {values}", self.font, self.MAX_HELP_TEXT_WIDTH)
                    totalTextHeight = len(wrappedText) * self.FONT_SIZE + (len(wrappedText) - 1) * (self.INPUT_TEXT_VERTICAL_SPACING * 0.3)  # Reduced spacing between lines
                    startY = inputBox.y + (self.INPUT_BOX_HEIGHT - totalTextHeight) / 2
    
                    # Calculate help text bounding box with right padding
                    helpTextX = inputBox.x + self.INPUT_BOX_WIDTH + self.INPUT_TEXT_HORIZONTAL_SPACING
                    helpTextY = startY
                    helpTextWidth = self.font.size(max(wrappedText, key=len))[0]
                    helpTextHeight = totalTextHeight
    
                    # Define the overall bounding rectangle with right padding
                    boundingRect = pygame.Rect(
                        inputBox.x - 2,  # Slight padding on the left
                        min(inputBox.y, helpTextY) - 2,  # Slight padding on the top
                        self.INPUT_BOX_WIDTH + self.INPUT_TEXT_HORIZONTAL_SPACING + helpTextWidth + 14,  # Increased width for right padding
                        max(self.INPUT_BOX_HEIGHT, helpTextHeight) + 4  # Height with padding
                    )
    
                    # Draw the overall contour around input box and help text
                    pygame.draw.rect(self.screen, self.TEXT_COLOR, boundingRect, 1)  # Single contour with width 1
    
                    # Draw input box with dynamic background color
                    pygame.draw.rect(self.screen, boxBackgroundColor, inputBox, 0)  # Filled input box with dynamic background
    
                    # Render input text
                    textSurface = self.font.render(str(self.inputValues[feature]), True, self.TEXT_COLOR)
                    self.screen.blit(textSurface, (inputBox.x + 5, inputBox.y + 5))
    
                    # Draw cursor
                    if inputBox == self.activeBox and self.cursorVisible:
                        self.cursor.topleft = (inputBox.x + 5 + textSurface.get_width(), inputBox.y + 5)
                        pygame.draw.rect(self.screen, self.TEXT_COLOR, self.cursor)  # Cursor should contrast with dark background
    
                    # Render help text
                    for j, line in enumerate(wrappedText):
                        labelSurface = self.font.render(line, True, self.TEXT_COLOR)
                        labelPos = (
                            helpTextX, 
                            helpTextY + j * (self.FONT_SIZE + self.INPUT_TEXT_VERTICAL_SPACING * 0.3)  # Reduced spacing between lines
                        )
                        self.screen.blit(labelSurface, labelPos)
    
                # Automatically get prediction
                inputData = [float(value) if self.isFloat(value) else np.nan for value in self.inputValues.values()]
                inputData = np.array([inputData])
                with self.predictionLock:
                    if not self.predictionReady:
                        self.predictionThread = threading.Thread(target=self.getPredictionThread)
                        self.predictionThread.start()
    
                prediction = None
                with self.predictionLock:
                    if self.prediction:
                        self.clearVisualizationArea()
                        prediction = self.prediction
                        intermediateOutputs = self.intermediateOutputs
                        self.prediction = None
                        self.intermediateOutputs = None
                        
                        if visualisation:
                            self.visualizeNeurons(self.screen, self.model, intermediateOutputs)

                if prediction:
                    # [0, 1] -> [-1, 1]
                    predictionConfidence = prediction[0][0] * 2 - 1
                    predictionText = "YES" if predictionConfidence > 0 else "NO"
                    predictionColor = self.interpolateColor(self.POSITIVE_COLOR, self.NEGATIVE_COLOR, prediction[0][0])
                    
                    # [-1, 0, 1] -> [100, 0, 100]
                    predictionConfidence = abs(predictionConfidence) * 100
                    predictionFont = pygame.font.Font(None, 25)
                    predictionSurface = predictionFont.render(f"{predictionText} ({predictionConfidence:.2f}%)", True, predictionColor)
    
                    predictionY = 100
                    predictionX = self.screen.get_width() - self.rightMargin - predictionSurface.get_width() - 10
                    self.screen.blit(predictionSurface, (predictionX, predictionY))

                    # Draw text above the visualization area
                    if self.enableClustering:
                        visualisationText = self.font.render("Neurons are visualized using clustering", True, self.TEXT_COLOR)
                        visualisationTextPos = (self.leftMargin, self.topMargin - 5)
                        self.screen.blit(visualisationText, visualisationTextPos)
    
            # Draw "Select Model" button only if no model is selected
            if not self.modelFilePath:
                self.selectModelButton = pygame.Rect(10, 30 * len(self.inputBoxes) + 60, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT)
                color = self.HOVER_COLOR if self.selectModelButton.collidepoint(pygame.mouse.get_pos()) else self.TEXT_COLOR
                pygame.draw.rect(self.screen, color, self.selectModelButton, 2)
                selectModelText = self.font.render("Select Model", True, color)
                self.screen.blit(selectModelText, (self.selectModelButton.x + 5, self.selectModelButton.y + 5))
    
           
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(self.fps)



def main():
    neuralNetVis = NeuralNetApp()
    neuralNetVis.mainLoop(True)

if __name__ == "__main__":
    main()