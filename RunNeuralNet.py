import os
import csv
import pygame
from tkinter import Tk, filedialog
from NeuralNet import predictWithModel

# Colors
NEGATIVE_COLOR = pygame.Color("#FD4F59")
POSITIVE_COLOR = pygame.Color("#5BAFFC")
TEXT_COLOR = pygame.Color("#DDDDDD")
BACKGROUND_COLOR = pygame.Color("#222222")

# Global variables for font size and input box size
FONT_SIZE = 15
INPUT_BOX_WIDTH = 140
INPUT_BOX_HEIGHT = 20
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800

# Load csv
def loadModelFeaturesMapping():
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

def getPrediction():
    inputData = []
    for feature, value in inputValues.items():
        if value.isdigit():
            inputData.append(int(value))
        else:
            inputData.append(value)
    prediction = predictWithModel(inputData, modelFilePath)
    return prediction

def visualizeNeurons(screen, layers):
    neuronRadius = 10
    xOffset = 400
    yOffset = 100
    layerSpacing = 150
    neuronSpacing = 50

    for i, layer in enumerate(layers):
        for j in range(layer):
            x = xOffset + i * layerSpacing
            y = yOffset + j * neuronSpacing
            pygame.draw.circle(screen, POSITIVE_COLOR if i == 0 else NEGATIVE_COLOR, (x, y), neuronRadius)

def selectModelFile():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.keras"), ("All Files", "*.*")])
    root.destroy()
    return file_path

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = []
    current_width = 0

    for word in words:
        word_width, _ = font.size(word + ' ')
        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width

    if current_line:
        lines.append(' '.join(current_line))

    return lines

def main():
    global inputValues, inputBoxes, mapping, modelFilePath

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neural Network Prediction")
    font = pygame.font.Font(None, FONT_SIZE)

    # Initialize variables
    inputBoxes = []
    inputValues = {}

    # Main loop
    running = True
    prediction = None
    activeBox = None
    activeFeature = None
    modelFilePath = None

    while running:
        screen.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if modelFilePath is None:
                    if selectModelButton.collidepoint(event.pos):
                        modelFilePath = selectModelFile()
                        if modelFilePath:
                            mapping = loadModelFeaturesMapping()
                            for i, (feature, values) in enumerate(mapping.items()):
                                inputBox = pygame.Rect(10, 30 * i + 10, INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT)
                                inputBoxes.append((feature, inputBox, values))
                                inputValues[feature] = ""
                else:
                    for feature, inputBox, values in inputBoxes:
                        if inputBox.collidepoint(event.pos):
                            activeBox = inputBox
                            activeFeature = feature
                    if 'predictDutton' in locals() and predictDutton.collidepoint(event.pos):
                        prediction = getPrediction()
            elif event.type == pygame.KEYDOWN:
                if activeBox:
                    if event.key == pygame.K_RETURN:
                        activeBox = None
                    elif event.key == pygame.K_BACKSPACE:
                        inputValues[activeFeature] = inputValues[activeFeature][:-1]
                    else:
                        inputValues[activeFeature] += event.unicode

        if modelFilePath:
            # Draw input fields
            for feature, inputBox, values in inputBoxes:
                pygame.draw.rect(screen, TEXT_COLOR, inputBox, 2)
                textSurface = font.render(inputValues[feature], True, TEXT_COLOR)
                screen.blit(textSurface, (inputBox.x + 5, inputBox.y + 5))
                
                # Wrap the feature help text
                wrapped_text = wrap_text(f"{feature}: {values}", font, 200)
                for j, line in enumerate(wrapped_text):
                    labelSurface = font.render(line, True, TEXT_COLOR)
                    screen.blit(labelSurface, (inputBox.x + inputBox.width + 10, inputBox.y + 5 + j * FONT_SIZE))

            # Draw prediction button
            predictDutton = pygame.Rect(10, 30 * len(inputBoxes) + 20, INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT)
            pygame.draw.rect(screen, TEXT_COLOR, predictDutton, 2)
            predictText = font.render("Predict", True, TEXT_COLOR)
            screen.blit(predictText, (predictDutton.x + 5, predictDutton.y + 5))

            # Display prediction
            if prediction:
                prediction_surface = font.render(f"Prediction: {prediction}", True, TEXT_COLOR)
                screen.blit(prediction_surface, (800, 10))

            # Visualize neurons
            visualizeNeurons(screen, [len(inputBoxes), 5, 2])  # Example layer structure

        # Draw select model button
        selectModelButton = pygame.Rect(10, 30 * len(inputBoxes) + 60, INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT)
        pygame.draw.rect(screen, TEXT_COLOR, selectModelButton, 2)
        selectModelText = font.render("Select Model", True, TEXT_COLOR)
        screen.blit(selectModelText, (selectModelButton.x + 5, selectModelButton.y + 5))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()