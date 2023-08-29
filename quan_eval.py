import warnings
import os
import numpy as np
import tensorflow as tf
from data_utils import load_rssi
from tensorflow import keras

data_directory = 'data_BLE/data/'
house_name = 'C'
window_size = 20
hop_size = 10
shuffle_data = False

X_train, y_train = load_rssi(data_directory, house_name, 'fp', shuffle_data, window_size, hop_size)
X_test, y_test = load_rssi(data_directory, house_name, 'fl', shuffle_data, window_size, hop_size)

APs = np.shape(X_train)[2]
NUM_CLASSES = len(set(y_train))
NUM_EPOCHS = 100
BATCH_SIZE = 64
model_type = '4conv'

# Load and compile the Keras model
model_path = 'model_path/custom_CNN_' + model_type + '_house_C.h5'
keras_model = keras.models.load_model(model_path)
optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_ = tf.keras.losses.SparseCategoricalCrossentropy()
keras_model.compile(optimizer=optimizer_, loss=loss_, metrics=['accuracy'])

# Convert the Keras model to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

# Save the quantized TFLite model
quantized_model_path = 'model_path/custom_CNN_' + model_type + '_house_C_quantized.tflite'
with open(quantized_model_path, 'wb') as f:
    f.write(quantized_tflite_model)

# Load the quantized TFLite model
# quantized_model_path = 'model_path/custom_CNN_' + model_type + '_house_C_int8.tflite'
interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tflite_accuracy = 0
predicted_labels = []
for n in range(len(X_test)):
    # preprocess input data
    # input_data = X_test[n].astype(np.int8)
    input_data = X_test[n].astype(np.float32)
    input_data = input_data[np.newaxis, ...]

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Perform inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_labels.append(np.argmax(output_data))
    tflite_accuracy += (np.argmax(output_data) == y_test[n])

tflite_accuracy /= len(X_test)
print("TFLite Model Accuracy:", tflite_accuracy * 100)

from sklearn.metrics import f1_score
macro_f1 = f1_score(y_test, predicted_labels, average='macro')
print(f"Macro F1 Score: {macro_f1*100:.2f}")