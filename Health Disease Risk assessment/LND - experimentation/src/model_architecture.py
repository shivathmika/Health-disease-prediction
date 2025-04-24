import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Flatten, Activation, BatchNormalization
)
from keras.optimizers import Adam
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.autograd import Variable
import skimage.transform


def buildCNNModel():
    # Define a convolutional neural network using Keras
    cnn_model = Sequential()
    
    # First convolutional block
    cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                         padding='same', input_shape=(64, 64, 1)))
    
    # Second convolutional layer
    cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                         padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Third convolutional block
    cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same'))
    cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                         padding='same'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Transition to dense layers
    cnn_model.add(GlobalAveragePooling2D())
    cnn_model.add(Flatten())
    
    # Output layer with softmax for classification
    cnn_model.add(Dense(units=2, activation='softmax'))
    
    print(cnn_model)
    
    return cnn_model
