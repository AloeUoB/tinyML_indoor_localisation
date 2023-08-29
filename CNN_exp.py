import warnings
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from data_utils import MinMaxScaler, windowing, load_rssi
from model_utils import create_simple_model, plot_train_history, plot_accuracy_history
from model_utils import convert_tflite_model, save_tflite_model, save_tf_model
from model_utils import create_custom_cnn
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('WARNING')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import sys
# sys.path.insert(0, '/data_BLE/')

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
model_type = 'custom_CNN_3conv'

# Example configuration
num_conv_layers = 3
num_neurons_per_layer = [32, 64, 128]  #[32, 64, 128, 256]  # Number of neurons for each convolution layer
pooling_sizes = [2, 2, 2]  # Pooling sizes for each MaxPooling layer
dense_neurons = 128

# Create the custom CNN model
model = create_custom_cnn(num_conv_layers, num_neurons_per_layer, pooling_sizes, dense_neurons,
                                 APs, window_size, NUM_CLASSES)
# Compile the model
optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_ = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer_, loss=loss_, metrics=['accuracy'])
# Print model summary
model.summary()
# Train the model
history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
# plot training and validation loss
plot_train_history(history, model_type)
plot_accuracy_history(history, model_type)

# Test the model
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, predicted_labels)
print("Test Accuracy:", test_accuracy*100)

# convert model to Tensor lite
tflite_model = convert_tflite_model(model)

# Save the trained model
save_directory = 'model_path'
save_tflite_model(tflite_model,
                  save_directory,
                  model_name=model_type+"_house_"+house_name+".tflite")

model_name = model_type + "_house_" + house_name + ".h5"
save_tf_model(model, save_directory, model_name)

