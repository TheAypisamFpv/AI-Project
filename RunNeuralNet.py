from NeuralNet import predictWithModel, loadModel
from tkinter import Tk, filedialog
import tensorflow as tf
import numpy as np
import threading
import difflib
import pygame
import csv
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
        self.FONT_SIZE = 15
        self.INPUT_BOX_WIDTH = 140
        self.INPUT_BOX_HEIGHT = 20
        self.MIN_WIDTH = 1000
        self.MIN_HEIGHT = 800
        self.TOP_BOTTOM_MARGIN = 10
        self.MAX_HELP_TEXT_WIDTH = 400  # Maximum width for input help text
        self.INPUT_TEXT_HORIZONTAL_SPACING = 20  # Initial horizontal spacing
        self.INPUT_TEXT_VERTICAL_SPACING = 5    # Initial vertical spacing

        # FPS setting
        self.fps = 60

        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Neural Network Prediction")
        self.screen = pygame.display.set_mode(
            (1400, 900), pygame.RESIZABLE
        )
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
        self.mapping = self.loadModelFeaturesMapping()

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
        self.MAX_CONNECTION_WIDTH = 7

    def loadModelFeaturesMapping(self):
        """
        Load the mapping values for the model features from a CSV file.
        """
        mapping = {}
        with open('GeneratedDataSet/MappingValues.csv', mode='r') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            values = next(reader)
            for header, value in zip(headers, values):
                if value.startswith("[") and value.endswith("]"):
                    value = eval(value)
                mapping[header] = value
        return mapping

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

        prediction, intermediateOutputs = self.getPrediction()
        with self.predictionLock:
            self.prediction = prediction
            self.intermediateOutputs = intermediateOutputs
            self.lastPrediction = prediction
            self.lastIntermediateOutputs = intermediateOutputs
            self.predictionReady = False

    def getPrediction(self):
        """
        Parse input values and get a prediction from the model.
        """
        inputData = []
        for inputName, inputValue in self.inputValues.items():
            if not inputValue:
                inputData.append(0)
                continue
            
            inputMapping = self.mapping[inputName]
            
            if isinstance(inputMapping[0], (int, float)):
                minValue, maxValue = inputMapping
                normalizedValue = (float(inputValue) - minValue) / (maxValue - minValue) * 2 - 1
                normalizedValue = max(-1, min(1, normalizedValue))
                inputData.append(normalizedValue)
                continue
            
            elif isinstance(inputMapping[0], str):
                inputMapping = [str(val).lower() for val in inputMapping]
                inputValue = inputValue.lower()
                closestMatch = difflib.get_close_matches(inputValue, inputMapping, n=1, cutoff=0.1)
                if closestMatch:
                    normalizedValue = (inputMapping.index(closestMatch[0]) / (len(inputMapping) - 1)) * 2 - 1
                    normalizedValue = max(-1, min(1, normalizedValue))
                    inputData.append(normalizedValue)
                    continue
            else:
                raise ValueError(f"Invalid data type for feature mapping: {inputMapping} (got {type(inputMapping)}, expected list)")

            inputData.append(0)

        print("Normalized input data:", inputData)  # Debug print
        prediction, intermediateOutputs = predictWithModel(self.model, np.array([inputData]))
        print("Prediction:", prediction)  # Debug print
        print("Intermediate outputs:", intermediateOutputs)  # Debug print
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

    def visualizeNeurons(self, screen, model, inputData, intermediateOutputs):
        """
        Create a visualization of the neural network with neurons displaying their output values,
        and outgoing connections colored based on the neuron's output value.
        """
        leftMargin = 600
        rightMargin = 150
        topMargin = 10
        bottomMargin = 10
    
        # Combine input data with intermediate outputs
        allOutputs = intermediateOutputs
    
        numLayers = len(allOutputs)
        maxNeurons = max([layerOutput.shape[1] for layerOutput in allOutputs])
    
        availableWidth = self.screen.get_width() - leftMargin - rightMargin
        layerSpacing = availableWidth / (numLayers - 1)
    
        availableHeight = self.screen.get_height() - topMargin - bottomMargin
        neuronSpacing = availableHeight / maxNeurons
    
        neuronRadius = max(5, int(min(layerSpacing, neuronSpacing) / 4))
    
        layers = [layerOutput.shape[1] for layerOutput in allOutputs]
    
        # Extract weights from the model
        weights = [layer.get_weights()[0] for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    
        # Normalize weights to a suitable range for line widths
        maxWeight = max([np.max(np.abs(w)) for w in weights])
        minWeight = min([np.min(np.abs(w)) for w in weights])
        weightRange = maxWeight - minWeight
    
        def normalizeWeight(weight):
            return self.MIN_CONNECTION_WIDTH + (self.MAX_CONNECTION_WIDTH - self.MIN_CONNECTION_WIDTH) * (np.abs(weight) - minWeight) / weightRange
    
        # First pass: Draw all neuron connections
        for i in range(len(layers) - 1):
            currentLayerSize = layers[i]
            nextLayerSize = layers[i + 1]
            x = leftMargin + i * layerSpacing
            nextX = leftMargin + (i + 1) * layerSpacing
    
            currentTotalLayerHeight = (currentLayerSize - 1) * neuronSpacing
            currentYStart = topMargin + (availableHeight - currentTotalLayerHeight) / 2
    
            nextTotalLayerHeight = (nextLayerSize - 1) * neuronSpacing
            nextYStart = topMargin + (availableHeight - nextTotalLayerHeight) / 2
    
            for j in range(currentLayerSize):
                y = currentYStart + j * neuronSpacing
                outputValue = allOutputs[i][0][j]
                normalizedOutput = (outputValue + 1) / 2
                color = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, normalizedOutput)
    
                for k in range(nextLayerSize):
                    nextY = nextYStart + k * neuronSpacing
                    weight = weights[i][j, k]
                    lineWidth = int(normalizeWeight(weight))
                    pygame.draw.line(screen, color, (int(x), int(y)), (int(nextX), int(nextY)), lineWidth)
    
        # Second pass: Draw all neurons and their values
        for i in range(len(layers)):
            layerSize = layers[i]
            x = leftMargin + i * layerSpacing
            totalLayerHeight = (layerSize - 1) * neuronSpacing
            yStart = topMargin + (availableHeight - totalLayerHeight) / 2
    
            for j in range(layerSize):
                y = yStart + j * neuronSpacing
                outputValue = allOutputs[i][0][j]
                normalizedOutput = (outputValue + 1) / 2
                color = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, normalizedOutput)
                pygame.draw.circle(screen, color, (int(x), int(y)), neuronRadius)
    
                # Render neuron output value
                valueText = f"{outputValue:.2f}"
                textSurface = self.font.render(valueText, True, self.TEXT_COLOR)
                textRect = textSurface.get_rect(center=(int(x), int(y)))
                screen.blit(textSurface, textRect)
 
    def mainLoop(self):
        while self.running:
            # Fill background
            self.screen.fill(self.BACKGROUND_COLOR)
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

                    # Update input box vertical positions based on new window height
                    if self.modelFilePath:
                        verticalSpacing = (self.screen.get_height() - 2 * self.TOP_BOTTOM_MARGIN) / len(self.mapping)
                        for i, (feature, inputBox, values) in enumerate(self.inputBoxes):
                            newY = self.TOP_BOTTOM_MARGIN + int(verticalSpacing * i)
                            self.inputBoxes[i] = (feature, pygame.Rect(10, newY, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT), values)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle mouse clicks
                    if self.modelFilePath is None:
                        # If no model is selected, check if "Select Model" button is clicked
                        if hasattr(self, 'selectModelButton') and self.selectModelButton.collidepoint(event.pos):
                            self.modelFilePath = self.selectModelFile()
                            if self.modelFilePath:
                                self.model = loadModel(self.modelFilePath)
                                self.mapping = self.loadModelFeaturesMapping()
                                numFeatures = len(self.mapping)
                                verticalSpacing = (self.screen.get_height() - 2 * self.TOP_BOTTOM_MARGIN) / numFeatures
                                for i, (feature, values) in enumerate(self.mapping.items()):
                                    inputBox = pygame.Rect(10, self.TOP_BOTTOM_MARGIN + int(verticalSpacing * i), self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT)
                                    self.inputBoxes.append((feature, inputBox, values))
                                    self.inputValues[feature] = ""
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
                
                # After getting inputData as a list
                inputData = [float(value) if self.isFloat(value) else np.nan for value in self.inputValues.values()]
                # Convert inputData to a NumPy array with shape (1, num_features)
                inputData = np.array([inputData])
                with self.predictionLock:
                    if not self.predictionReady:
                        self.predictionThread = threading.Thread(target=self.getPredictionThread)
                        self.predictionThread.start()

                with self.predictionLock:
                    if self.prediction is not None:
                        prediction = self.prediction
                        intermediateOutputs = self.intermediateOutputs
                        self.prediction = None
                        self.intermediateOutputs = None
                        if prediction is None:
                            predictionText = "N/A"
                            predictionSurface = self.font.render(f"Prediction: {predictionText} ({predictionText})", True, self.TEXT_COLOR)
                        else:
                            predictionText = "YES" if prediction[0][0] > 0.5 else "NO"
                            sqrtPrediction = np.sqrt(prediction[0][0])
                            predictionConfidence = abs((sqrtPrediction - 0.5) *200)
                            predictionSurface = self.font.render(f"Prediction: {predictionText} ({predictionConfidence:.2f}%)", True, self.TEXT_COLOR)
                            self.visualizeNeurons(self.screen, self.model, inputData, intermediateOutputs)

                        predictionY = (self.screen.get_height() - predictionSurface.get_height()) / 2
                        self.screen.blit(predictionSurface, (self.screen.get_width() - predictionSurface.get_width() - 10, predictionY))
                    
                    elif self.lastPrediction is not None:
                        prediction = self.lastPrediction
                        intermediateOutputs = self.lastIntermediateOutputs
                        if prediction is None:
                            predictionText = "N/A"
                            predictionSurface = self.font.render(f"Prediction: {predictionText} ({predictionText})", True, self.TEXT_COLOR)
                        else:
                            predictionText = "YES" if prediction[0][0] > 0.5 else "NO"
                            sqrtPrediction = np.sqrt(prediction[0][0])
                            predictionConfidence = abs((sqrtPrediction - 0.5) *200)
                            predictionSurface = self.font.render(f"Prediction: {predictionText} ({predictionConfidence:.2f}%)", True, self.TEXT_COLOR)
                            self.visualizeNeurons(self.screen, self.model, inputData, intermediateOutputs)

                        predictionY = (self.screen.get_height() - predictionSurface.get_height()) / 2
                        self.screen.blit(predictionSurface, (self.screen.get_width() - predictionSurface.get_width() - 10, predictionY))

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

    def isFloat(self, value):
        """Check if the string can be converted to a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

def main():
    neuralNetVis = NeuralNetApp()
    neuralNetVis.mainLoop()

if __name__ == "__main__":
    main()