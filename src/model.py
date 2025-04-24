import tensorflow as tf
import keras
from keras import models, Input
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense

def get_model(input_shape, num_classes: int, print_summary=False):
    """
    Get sequential model

    Parameters:
    ----------
    input_shape: Shape of input image
    num_classes: The number of output classes
    print_summary: Prints out model architecture discription when true
    """
    model = models.Sequential([

        Input(shape=(input_shape)),

        Conv2D(128, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),

        Conv2D(256, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),

        Conv2D(256, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),

        Conv2D(256, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),

        Conv2D(256, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),
        
        Flatten(),

        Dense(512, activation='relu'),
        Dense(256, activation='relu'),

        Dense(num_classes, activation='softmax')
    ])

    if print_summary:
        model.summary()
    
    return model