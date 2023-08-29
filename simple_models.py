import warnings
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from data_utils import MinMaxScaler, windowing
from model_utils import create_simple_model, plot_train_history, plot_accuracy_history
from model_utils import convert_tflite_model, save_tflite_model, save_tf_model
warnings.filterwarnings("ignore")
# import sys
# sys.path.insert(0, '/data_BLE/')

data_directory = 'data_BLE/data/'
house_name = 'C'

# load fingerprint data
datatype = 'fp'
house_file = 'csv_house_' + house_name + '_' + datatype + '.csv'
X_train = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=list(range(1, 12)))
y_train = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[12])
y_train = y_train - 1
X_train, min_val, max_val = MinMaxScaler(X_train)
# windowing
# X_train, y_train = windowing(X_train, y_train, seq_len=20, hop_size=10, shuffle=False)
# X_train = np.transpose(X_train, (0, 2, 1))
# X_train = np.reshape(X_train, (-1, 220))

# load free living data
datatype = 'fl'
house_file = 'csv_house_' + house_name + '_' + datatype + '.csv'
X_test = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=list(range(1, 12)))
y_test = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[12])
y_test = y_test-1
X_test, min_val, max_val = MinMaxScaler(X_test)
# windowing
# X_test, y_test = windowing(X_test, y_test, seq_len=20, hop_size=10, shuffle=False)
# X_test = np.transpose(X_test, (0, 2, 1))
# X_test = np.reshape(X_test, (-1, 220))

NUM_EPOCHS = 100
BATCH_SIZE = 64

APs = 11
NUM_CLASSES = len(np.unique(y_train))

model_type_all = ["MLP"]  #["MLP", "CNN", "LSTM"]
for model_type in model_type_all:
    if model_type is not "MLP":
        X_train = X_train.reshape(len(X_train), 11, 1)
        X_test = X_test.reshape(len(X_test), 11, 1)

    # create localisation model
    model = create_simple_model(model_type, NUM_CLASSES)
    # layers_data = [(0, 64, 0.5), (1, 256, 0.4)]
    # model = MLP(APs*1, layers_data, NUM_CLASSES)
    # model.build(input_shape=(None, APs * 1))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # lot training and validation loss
    plot_train_history(history, model_type)
    plot_accuracy_history(history, model_type)

    # Test the model
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    # predictions = np.round(predictions).flatten() # Convert predictions to integer labels if needed

    # Calculate test accuracy
    # test_accuracy = accuracy_score(y_test, predictions)
    test_accuracy = accuracy_score(y_test, predicted_labels)
    print("Test Accuracy:", test_accuracy)

    # convert model to Tensor lite
    tflite_model = convert_tflite_model(model)

    # Save the trained model
    save_directory = 'model_path'
    save_tflite_model(tflite_model,
                      save_directory,
                      model_name="trained_simple_"+model_type+"_model_house_"+house_name+".tflite")

    model_name = "trained_" + model_type + "_model_house_" + house_name + ".h5"
    save_tf_model(model, save_directory, model_name)




