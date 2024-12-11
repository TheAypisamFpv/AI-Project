# AI Project
### Big Picture
- sanitize dataset
- seperate dataset into training and testing
- train model on training dataset
- test model on testing dataset
- visualize/validation model
---
### Model
- Linear Regression
---
### Model Visualization
- Each point on the map is a house, with the color representing the difference between the predicted price and the actual price.
- The color is on a scale from a red to a green.
- The scale for the color is normalized and used in this function: 

![f(x) = \frac{10^x}{10} \cdot x](https://latex.codecogs.com/png.latex?\color{white}f(x)%20=%20\frac{10^x}{10}%20\cdot%20x)

- Then animated over time to show the progression of the model.


## Requirements
### Python 3.11.x - 3.12.x

Python can be downloaded from the official website: https://www.python.org/downloads/

- Create a virtual environment using the installed python version
- Install the required packages using the following command:

```bash
pip install -r requirements.txt
```