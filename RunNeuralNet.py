import os
import csv
import pygame
from NeuralNet import predictWithModel

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

mapping = loadModelFeaturesMapping()

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1200, 600))
pygame.display.set_caption("Neural Network Prediction")
font = pygame.font.Font(None, 24)

# Colors
NEGATIVE_COLOR = pygame.Color("#FD4F59")
POSITIVE_COLOR = pygame.Color("#5BAFFC")
TEXT_COLOR = pygame.Color("#DDDDDD")
BACKGROUND_COLOR = pygame.Color("#222222")

# Input fields
inputBoxes = []
inputValues = {}
for i, (feature, values) in enumerate(mapping.items()):
    inputBox = pygame.Rect(10, 30 * i + 10, 140, 24)
    inputBoxes.append((feature, inputBox, values))
    inputValues[feature] = ""

# Function to get the prediction
def getPrediction():
    inputData = []
    for feature, value in inputValues.items():
        if value.isdigit():
            inputData.append(int(value))
        else:
            inputData.append(value)
    prediction = predictWithModel(inputData)
    return prediction

# Function to visualize the neurons
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

# Main loop
running = True
prediction = None
activeBox = None
activeFeature = None
while running:
    screen.fill(BACKGROUND_COLOR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            for feature, inputBox, values in inputBoxes:
                if inputBox.collidepoint(event.pos):
                    activeBox = inputBox
                    activeFeature = feature
        elif event.type == pygame.KEYDOWN:
            if activeBox:
                if event.key == pygame.K_RETURN:
                    activeBox = None
                elif event.key == pygame.K_BACKSPACE:
                    inputValues[activeFeature] = inputValues[activeFeature][:-1]
                else:
                    inputValues[activeFeature] += event.unicode

    # Draw input fields
    for feature, inputBox, values in inputBoxes:
        pygame.draw.rect(screen, TEXT_COLOR, inputBox, 2)
        text_surface = font.render(inputValues[feature], True, TEXT_COLOR)
        screen.blit(text_surface, (inputBox.x + 5, inputBox.y + 5))
        label_surface = font.render(f"{feature}: {values}", True, TEXT_COLOR)
        screen.blit(label_surface, (inputBox.x + inputBox.width + 10, inputBox.y + 5))

    # Draw prediction button
    predict_button = pygame.Rect(10, 30 * len(inputBoxes) + 20, 140, 24)
    pygame.draw.rect(screen, TEXT_COLOR, predict_button, 2)
    predict_text = font.render("Predict", True, TEXT_COLOR)
    screen.blit(predict_text, (predict_button.x + 5, predict_button.y + 5))

    if event.type == pygame.MOUSEBUTTONDOWN and predict_button.collidepoint(event.pos):
        prediction = getPrediction()

    # Display prediction
    if prediction:
        prediction_surface = font.render(f"Prediction: {prediction}", True, TEXT_COLOR)
        screen.blit(prediction_surface, (800, 10))

    # Visualize neurons
    visualizeNeurons(screen, [len(inputBoxes), 5, 2])  # Example layer structure

    pygame.display.flip()

pygame.quit()