import warnings
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from data_utils import MinMaxScaler, windowing, load_rssi
from model_utils import create_simple_model, plot_train_history, plot_accuracy_history
from model_utils import convert_tflite_model, save_tflite_model, save_tf_model
from model_utils import create_custom_cnn, create_custom_mlp, create_custom_lstm
import wandb
import yaml
import argparse

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('WARNING')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import sys
# sys.path.insert(0, '/data_BLE/')

def get_layer_pool_size():
    all_layer_size = [wandb.config.layer_size_1,
                      wandb.config.layer_size_2,
                      wandb.config.layer_size_3,
                      wandb.config.layer_size_4,
                      wandb.config.layer_size_5,
                      wandb.config.layer_size_6,
                      wandb.config.layer_size_7,
                      wandb.config.layer_size_8,
                      wandb.config.layer_size_9,
                      wandb.config.layer_size_10,
                      ]
    all_pool_size = [wandb.config.pool_size_2,
                     wandb.config.pool_size_3,
                     wandb.config.pool_size_4,
                     wandb.config.pool_size_5,
                     wandb.config.pool_size_6,
                     wandb.config.pool_size_7,
                     wandb.config.pool_size_8,
                     wandb.config.pool_size_9,
                     wandb.config.pool_size_10,
                       ]
    layer_size = []
    pool_size = []
    num_layers = wandb.config.num_layers
    for i in range(num_layers):
        layer_size.append(all_layer_size[i])
        pool_size.append(all_pool_size[i])

    return layer_size, pool_size

    print(num_layers)
    print(layer_size)
    print(pool_size)

def get_layer_size():
    all_layer_size = [wandb.config.layer_size_1,
                      wandb.config.layer_size_2,
                      wandb.config.layer_size_3,
                      wandb.config.layer_size_4,
                      wandb.config.layer_size_5,
                      wandb.config.layer_size_6,
                      wandb.config.layer_size_7,
                      wandb.config.layer_size_8,
                      wandb.config.layer_size_9,
                      wandb.config.layer_size_10,
                      ]
    layer_size = []
    num_layers = wandb.config.num_layers
    for i in range(num_layers):
        layer_size.append(all_layer_size[i])

    return layer_size

def main():
    data_directory = 'data_BLE/data/'
    house_name = 'C'
    window_size = 20
    hop_size = round(window_size/2)
    shuffle_data = False

    X_train, y_train = load_rssi(data_directory, house_name, 'fp', shuffle_data, window_size, hop_size)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
    APs = np.shape(X_train)[2]
    NUM_CLASSES = len(set(y_train))

    with open('tinyML_indoor_localisation/config_wandb.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    NUM_EPOCHS = 100
    BATCH_SIZE = wandb.config.batch_size
    LEARN_RATE = wandb.config.learning_rate
    model_option = 'lstm'     #wandb.config.model_option

    # Example configuration
    num_conv_layers = wandb.config.num_layers
    # num_neurons_per_layer, pooling_sizes = get_layer_pool_size()
    num_neurons_per_layer= get_layer_size()
    pooling_sizes = []
    dense_neurons = 128

    # Create the custom CNN model
    if model_option == 'mlp':
        model = create_custom_mlp(num_conv_layers, num_neurons_per_layer, pooling_sizes, dense_neurons,
                                         APs, window_size, NUM_CLASSES)
    elif model_option == 'cnn':
        model = create_custom_cnn(num_conv_layers, num_neurons_per_layer, pooling_sizes, dense_neurons,
                                  APs, window_size, NUM_CLASSES)
    elif model_option == 'lstm':
        model = create_custom_lstm(num_conv_layers, num_neurons_per_layer, pooling_sizes, dense_neurons,
                                  APs, window_size, NUM_CLASSES)
    # Compile the model
    optimizer_ = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer_, loss=loss_, metrics=['accuracy'])
    # Print model summary
    model.summary()

    # Train the model
    for epoch in range(wandb.config.epochs):
        history = model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE, validation_split=0.2)

        # predictions_train = model.predict(X_train)
        # predicted_labels_train= np.argmax(predictions_train, axis=1)
        # macro_f1_train = f1_score(y_train, predicted_labels, average='macro')

        wandb.log({"epoch": epoch,
                   "loss": history.history['loss'][0],
                   "accuracy": history.history['accuracy'][0],
                   "val_accuracy": history.history['val_accuracy'][0]})
    # Test the model
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    test_accuracy = accuracy_score(y_test, predicted_labels)
    macro_f1 = f1_score(y_test, predicted_labels, average='macro')
    wandb.log({"test_accuracy": test_accuracy, "macro_f1": macro_f1})

main()
