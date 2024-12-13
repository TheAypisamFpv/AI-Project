import os
import csv
import pygame
from tkinter import Tk, filedialog
from NeuralNet import predictWithModel, loadModel
import numpy as np
import tensorflow as tf
import threading

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
        self.prediction_thread = None
        self.prediction_lock = threading.Lock()
        self.prediction_ready = False

        # Store last valid prediction
        self.last_prediction = None
        self.last_intermediateOutputs = None

        # Variables for text selection
        self.selected_text = ""
        self.selection_start = None
        self.selection_end = None

    def loadModelFeaturesMapping(self):
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
        root = Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.keras"), ("All Files", "*.*")])
        root.destroy()
        return file_path

    def wrapText(self, text, font, maxWidth):
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
        with self.prediction_lock:
            if self.prediction_ready:
                return
            self.prediction_ready = True

        prediction, intermediateOutputs = self.getPrediction()
        with self.prediction_lock:
            self.prediction = prediction
            self.intermediateOutputs = intermediateOutputs
            self.last_prediction = prediction
            self.last_intermediateOutputs = intermediateOutputs
            self.prediction_ready = False

    def getPrediction(self):
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
                if inputValue in inputMapping:
                    normalizedValue = (inputMapping.index(inputValue) / (len(inputMapping) - 1)) * 2 - 1
                    normalizedValue = max(-1, min(1, normalizedValue))
                    inputData.append(normalizedValue)
                    continue
            else:
                raise ValueError(f"Invalid data type for feature mapping: {inputMapping} (got {type(inputMapping)}, expected list)")

            inputData.append(0)

        print("Input data:", inputData)
        prediction, intermediateOutputs = predictWithModel(self.model, np.array([inputData]))
        print("Prediction:", prediction)
        return prediction, intermediateOutputs

    def interpolateColor(self, colorA, colorB, factor:float):
        """Interpolate between two colors with a given factor (0 to 1)."""
        return pygame.Color(
            int(colorA.r + (colorB.r - colorA.r) * factor),
            int(colorA.g + (colorB.g - colorA.g) * factor),
            int(colorA.b + (colorB.b - colorA.b) * factor)
        )

    def visualizeNeurons(self, screen, model, inputData, intermediate_outputs):
        """
        Create a visualization of the neural network.
        """
        leftMargin = 600
        rightMargin = 150
        topMargin = 10
        bottomMargin = 10

        numLayers = len(model.layers)
        maxNeurons = max([layer.units for layer in model.layers if hasattr(layer, 'units')])

        availableWidth = self.screen.get_width() - leftMargin - rightMargin
        layerSpacing = availableWidth / (numLayers - 3)

        availableHeight = self.screen.get_height() - topMargin - bottomMargin
        neuronSpacing = availableHeight / maxNeurons

        neuronRadius = max(5, int(min(layerSpacing, neuronSpacing) / 4))

        layers = [layer.units for layer in model.layers if hasattr(layer, 'units')]

        # First pass: Draw all neuron connections
        for i, layerSize in enumerate(layers):
            x = leftMargin + i * layerSpacing
            totalLayerHeight = (layerSize - 1) * neuronSpacing
            yStart = topMargin + (availableHeight - totalLayerHeight) / 2

            for j in range(layerSize):
                y = yStart + j * neuronSpacing

                if i < len(layers) - 1:
                    nextLayerSize = layers[i + 1]
                    nextTotalLayerHeight = (nextLayerSize - 1) * neuronSpacing
                    nextYStart = topMargin + (availableHeight - nextTotalLayerHeight) / 2
                    for k in range(nextLayerSize):
                        nextX = leftMargin + (i + 1) * layerSpacing
                        nextY = nextYStart + k * neuronSpacing
                        nextOutputValue = max(0.0, min(1.0, intermediate_outputs[i + 1][0][k]))
                        nextColor = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, nextOutputValue)
                        pygame.draw.line(screen, nextColor, (int(x), int(y)), (int(nextX), int(nextY)), 1)

        # Second pass: Draw all neurons and their values
        for i, layerSize in enumerate(layers):
            x = leftMargin + i * layerSpacing
            totalLayerHeight = (layerSize - 1) * neuronSpacing
            yStart = topMargin + (availableHeight - totalLayerHeight) / 2

            for j in range(layerSize):
                y = yStart + j * neuronSpacing
                outputValue = max(0.0, min(1.0, intermediate_outputs[i][0][j]))
                color = self.interpolateColor(self.NEGATIVE_COLOR, self.POSITIVE_COLOR, outputValue)
                pygame.draw.circle(screen, color, (int(x), int(y)), neuronRadius)

                # Render neuron value
                value_text = f"{outputValue:.2f}"
                text_surface = self.font.render(value_text, True, self.TEXT_COLOR)
                text_rect = text_surface.get_rect(center=(int(x), int(y)))
                screen.blit(text_surface, text_rect)

    def main_loop(self):
        while self.running:
            # Fill background
            self.screen.fill(self.BACKGROUND_COLOR)
            self.cursorTimer += 1

            # Toggle cursor visibility
            if self.cursorTimer % (self.fps // 2) == 0:
                self.cursorVisible = not self.cursorVisible

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    new_width = max(event.w, self.MIN_WIDTH)
                    new_height = max(event.h, self.MIN_HEIGHT)
                    self.screen = pygame.display.set_mode(
                        (new_width, new_height), pygame.RESIZABLE
                    )
                    # Update input text spacing based on new window height
                    self.INPUT_TEXT_HORIZONTAL_SPACING = new_height * 0.02  # 2% of window height
                    self.INPUT_TEXT_VERTICAL_SPACING = new_height * 0.005    # 0.5% of window height

                    # Update input box vertical positions based on new window height
                    if self.modelFilePath:
                        verticalSpacing = (self.screen.get_height() - 2 * self.TOP_BOTTOM_MARGIN) / len(self.mapping)
                        for i, (feature, inputBox, values) in enumerate(self.inputBoxes):
                            new_y = self.TOP_BOTTOM_MARGIN + int(verticalSpacing * i)
                            self.inputBoxes[i] = (feature, pygame.Rect(10, new_y, self.INPUT_BOX_WIDTH, self.INPUT_BOX_HEIGHT), values)
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
                    box_background_color = self.ACTIVE_COLOR if inputBox == self.activeBox else self.INPUT_BACKGROUND_COLOR

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
                    pygame.draw.rect(self.screen, box_background_color, inputBox, 0)  # Filled input box with dynamic background

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
                inputData = [float(value) if self.is_float(value) else np.nan for value in self.inputValues.values()]
                with self.prediction_lock:
                    if not self.prediction_ready:
                        self.prediction_thread = threading.Thread(target=self.getPredictionThread)
                        self.prediction_thread.start()

                with self.prediction_lock:
                    if self.prediction is not None:
                        prediction = self.prediction
                        intermediateOutputs = self.intermediateOutputs
                        self.prediction = None
                        self.intermediateOutputs = None
                        if prediction is None:
                            predictionText = "N/A"
                            predictionSurface = self.font.render(f"Prediction: {predictionText} ({predictionText})", True, self.TEXT_COLOR)
                        else:
                            predictionText = "YES" if prediction[0][0] > 0 else "NO"
                            predictionSurface = self.font.render(f"Prediction: {predictionText} ({prediction[0][0]:.2f})", True, self.TEXT_COLOR)
                            self.visualizeNeurons(self.screen, self.model, inputData, intermediateOutputs)

                        predictionY = (self.screen.get_height() - predictionSurface.get_height()) / 2
                        self.screen.blit(predictionSurface, (self.screen.get_width() - predictionSurface.get_width() - 10, predictionY))
                    elif self.last_prediction is not None:
                        prediction = self.last_prediction
                        intermediateOutputs = self.last_intermediateOutputs
                        if prediction is None:
                            predictionText = "N/A"
                            predictionSurface = self.font.render(f"Prediction: {predictionText} ({predictionText})", True, self.TEXT_COLOR)
                        else:
                            predictionText = "YES" if prediction[0][0] > 0 else "NO"
                            predictionSurface = self.font.render(f"Prediction: {predictionText} ({prediction[0][0]:.2f})", True, self.TEXT_COLOR)
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

    def is_float(self, value):
        """Check if the string can be converted to a float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

def main():
    app = NeuralNetApp()
    app.main_loop()

if __name__ == "__main__":
    main()