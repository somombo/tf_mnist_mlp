# Example Tensorflow MNIST Multi Layer Perceptron Network

## Installation

Install Python / Tensorflow and supporting libraries. We recommend doing this through anaconda/miniconda [See here](https://www.tensorflow.org/install/).


## Execution 
Once installed you can run:
```
python ./main.py
```

## Usage
This automatically downloads  the MNIST dataset and then uses to Tensorflow to try and learn the parameters of a three layer network (in, hidden and out).

Main code is in `main.py`. Feel free to play around with the following parameters:

```python

LAMBDA = 1
HIDDEN_UNITS = 1024
EPOCHS = 10
ALPHA = 0.08
IMAGE_COLOR = 'bwr' # see matplotlib `./color_maps_labelled.png`
DISPLAY_EVERY = 1
```

The directory `saved_params` contains saved parameters (i.e weights and biases) from previous trainings. Feel free to delete or rename this if you would like to start from scratch with randomly initialized parameters.
