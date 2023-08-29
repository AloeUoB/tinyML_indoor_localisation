import numpy as np
from data_utils import MinMaxScaler, windowing
from model_utils import create_simple_model, plot_train_history, plot_accuracy_history, CustomModel

data_directory = 'data_BLE/data/'
house_name = 'C'

# load fingerprint data
datatype = 'fp'
house_file = 'csv_house_' + house_name + '_' + datatype + '.csv'
X_train = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=list(range(1, 12)))
y_train = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[12])
X_train, min_val, max_val = MinMaxScaler(X_train)
y_train = y_train-1
# windowing
X_train, y_train = windowing(X_train, y_train, seq_len=20, hop_size=10, shuffle=False)
X_train = np.reshape(X_train, (-1, 220))

# load free living data
datatype = 'fl'
house_file = 'csv_house_' + house_name + '_' + datatype + '.csv'
X_test = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=list(range(1, 12)))
y_test = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[12])
X_test, min_val, max_val = MinMaxScaler(X_test)
y_test = y_test-1
# windowing
X_test, y_test = windowing(X_test, y_test, seq_len=20, hop_size=10, shuffle=False)
X_test = np.reshape(X_test, (-1, 220))

NUM_EPOCHS = 100
BATCH_SIZE = 64
APs = 11
NUM_CLASSES = len(np.unique(y_train))

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_size, layers_data: list, output_size):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.layers.append(nn.Flatten())

        for idx, size, dropout in layers_data:
            if idx != layers_data[-1][0]:
                self.layers.append(nn.Linear(input_size, size))
                input_size = size
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
            else:
                self.layers.append(nn.Linear(input_size, output_size))
                self.layers.append(nn.Softmax(dim=1))

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

layers_data = [(0, 64, 0.5), (1, 256, 0.4)]
model = MLP(APs * 20, layers_data, NUM_CLASSES)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0067)



# model = CustomModel()
# criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss
# optimizer = optim.Adam(model.parameters(), lr=0.001)
X_train_tensor = torch.tensor(X_train).to(torch.float32)
y_train_tensor = torch.tensor(y_train).to(torch.long)
# X_train_tensor = X_train_tensor.to(torch.float32)
# y_train_tensor = y_train_tensor.to(torch.long)

X_test_tensor = torch.tensor(X_test).to(torch.float32)
y_test_tensor = torch.tensor(y_test).to(torch.long)
# X_test_tensor = X_test_tensor.to(torch.float32)
# y_test_tensor = y_test_tensor.to(torch.long)


# y_train_tensor = y_train_tensor.view(-1, 1)
# y_test_tensor = y_test_tensor.view(-1, 1)


# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted_labels = torch.argmax(outputs, dim=1)
    accuracy = (predicted_labels == y_test_tensor).float().mean()

print("Test Accuracy:", accuracy.item())

from sklearn.metrics import f1_score
macro_f1 = f1_score(y_test_tensor, predicted_labels, average='macro')

print(f"Macro F1 Score: {macro_f1:.2f}")
