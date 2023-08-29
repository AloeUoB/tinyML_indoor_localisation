import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense, ReLU, Dropout, Softmax, Input
import os
from absl import logging
import matplotlib.pyplot as plt



import torch.nn as nn
# Define the model class
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(11, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


import tensorflow as tf
from tensorflow.keras import layers, models

def create_custom_cnn(num_conv_layers, num_neurons_per_layer, pooling_sizes, dense_neurons, APs, window, num_classes):
    assert len(
        num_neurons_per_layer) == num_conv_layers, "Number of neurons per layer should match the number of convolution layers"
    assert len(
        pooling_sizes) == num_conv_layers, "Number of pooling sizes should match the number of convolution layers"

    model = models.Sequential()

    # Add convolutional and pooling layers
    for i in range(num_conv_layers):
        if i == 0:
            model.add(layers.Conv1D(num_neurons_per_layer[i], 1, activation='relu', input_shape=(window, APs)))
        else:
            model.add(layers.Conv1D(num_neurons_per_layer[i], 1, activation='relu'))

        model.add(layers.MaxPooling1D(pooling_sizes[i]))
    model.add(layers.Flatten())

    # Add dense layers
    # model.add(layers.Dense(dense_neurons, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def create_custom_mlp(num_conv_layers, num_neurons_per_layer, pooling_sizes, dense_neurons, APs, window, num_classes):
    assert len(
        num_neurons_per_layer) == num_conv_layers, "Number of neurons per layer should match the number of convolution layers"
    # assert len(
    #     pooling_sizes) == num_conv_layers, "Number of pooling sizes should match the number of convolution layers"

    model = models.Sequential()

    # Add convolutional and pooling layers
    for i in range(num_conv_layers):
        if i == 0:
            model.add(Input(shape=(window, APs)))
            model.add(Flatten())
            model.add(layers.Dense(num_neurons_per_layer[i], activation='relu'))
            model.add(layers.Dropout(0.5))
        else:
            model.add(layers.Dense(num_neurons_per_layer[i], activation='relu'))

        # model.add(layers.MaxPooling1D(pooling_sizes[i]))
    # model.add(layers.Flatten())

    # Add dense layers
    # model.add(layers.Dense(dense_neurons, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def create_custom_lstm(num_conv_layers, num_neurons_per_layer, pooling_sizes, dense_neurons, APs, window, num_classes):
    assert len(
        num_neurons_per_layer) == num_conv_layers, "Number of neurons per layer should match the number of convolution layers"

    model = models.Sequential()

    # Add convolutional and pooling layers
    for i in range(num_conv_layers):
        if i == 0:
            model.add(layers.LSTM(num_neurons_per_layer[i],input_shape=(window, APs), return_sequences=True))
        elif i == num_conv_layers:
            model.add(layers.LSTM(num_neurons_per_layer[i], return_sequences=False))
        else:
            model.add(layers.LSTM(num_neurons_per_layer[i]))

    # Add dense layers
    # model.add(layers.Dense(dense_neurons, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def create_simple_model(model_type, number_classes):
    """ Create the smallest model for each model type"""
    if model_type == "MLP":
        model = Sequential([
            Input(shape=(11, 1)),
            Flatten(),
            Dense(64, activation='relu'),
            # Dense(32, activation='relu'),
            Dense(number_classes, activation='softmax')
        ])
        # model = Sequential([
        #     Dense(128, activation='relu', input_shape=(11,)),
        #     Dense(64, activation='relu'),
        #     Dense(32, activation='relu'),
        #     Dense(16, activation='relu'),
        #     Dense(number_classes, activation='softmax')
        # ])


    elif model_type == "CNN":
        model = Sequential([
            Conv1D(16, kernel_size=3, activation='relu', input_shape=(11, 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(number_classes, activation='softmax')
        ])
    elif model_type == "LSTM":
        model = Sequential([
            LSTM(8, input_shape=(11, 1)),
            Dense(number_classes, activation='softmax')
        ])
    else:
        raise ValueError("Invalid model type")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def convert_tflite_model(model):
  """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
    Args:
        model (tf.keras.Model): the trained hello_world Model
    Returns:
        The converted model in serialized format.
  """
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
  """save the converted tflite model
  Args:
      tflite_model (binary): the converted model in serialized format.
      save_dir (str): the save directory
      model_name (str): model name to be saved
  """
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, model_name)
  with open(save_path, "wb") as f:
    f.write(tflite_model)
  logging.info("Tflite model saved to %s", save_dir)

def save_tf_model(model, save_directory, model_name):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model_path = os.path.join(save_directory, model_name)
    model.save(model_path)
    print("Model saved as", model_name)

def plot_accuracy_history(train_history, model_name):
    plt.plot(train_history.history['accuracy'], label='Train Accuracy')
    plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy: '+ model_name)
    plt.show()

def plot_train_history(train_history, model_name):
    """ Plot training and validation loss """
    plt.plot(train_history.history['loss'], label='train loss')
    plt.plot(train_history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss: '+ model_name)
    plt.show()


class MLP(tf.keras.Model):
    def __init__(self, input_size, layer_configs, output_size):
        super().__init__()

        self.custom_layers = []  # Use a different attribute name
        self.input_size = input_size
        self.custom_layers.append(Flatten(input_shape=(input_size,)))

        for idx, size, dropout in layer_configs:
            if idx != layer_configs[-1][0]:
                self.custom_layers.append(Dense(size))
                self.custom_layers.append(ReLU())
                self.custom_layers.append(Dropout(dropout))
            else:
                self.custom_layers.append(Dense(output_size))
                self.custom_layers.append(Softmax())

    def call(self, inputs):
        for layer in self.custom_layers:
            inputs = layer(inputs)
        return inputs

