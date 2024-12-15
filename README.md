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


# Neural Network
### Best models as of now:
#### With `AverageHoursWorked` (Accuracy:0.95):

`TrainedModel_[25, 512, 256, 128, 1]_100_32_0.3_0.001_relu_relu_sigmoid_Accuracy_mean_squared_error_Adam(0.0005)_0.2`


#### Without `AverageHoursWorked` (Accuracy:0.93):

`TrainedModel_[24, 256, 128, 64, 1]_100_20_0.3_0.001_relu_tanh_sigmoid_Accuracy_mean_squared_error_Adam(0.001)_0.2`


can be found in the `models` directory in the folder of the same name.
