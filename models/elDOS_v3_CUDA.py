import numpy as np
import argparse
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='Loading dataset.')

# Add arguments for channels and datas
parser.add_argument('--channels', type=int, required=True, help='Number of channels')
parser.add_argument('--id', type=int, required=True, help='Mission ID')

# Parse the arguments
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOSs = np.load('dataset/DOS_4000.npy')
fixed_len = len(DOSs[0])

rdf_2d = np.load('dataset/rdfs_2d.npy')
rdf_3d = np.load('dataset/rdfs_3d.npy')
adf_raw = np.load('dataset/adf_no_pbc.npy')

if len(adf_raw[0]) < len(rdf_2d[0]):
    adf = np.zeros_like(rdf_2d)
    for i in range(len(adf_raw)):
        adf[i][:len(adf_raw[i])] = adf_raw[i]

print(torch.cuda.is_available(), torch.__version__)

# zero padding to adf
rdf = np.stack([rdf_2d, rdf_3d, adf][:args.channels], axis=1)
channels = args.channels # 3
print(rdf.shape, channels)

rdf = np.log(rdf + 1e-4)

# Convert to PyTorch tensors
rdf = torch.tensor(rdf, dtype=torch.float32).to(device)
DOSs = torch.tensor(DOSs, dtype=torch.float32).to(device) * 10

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(rdf, DOSs, test_size=0.2)#, random_state=666)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)#, random_state=666)

# Create TensorDatasets and DataLoaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
valid_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)
valid_loader = DataLoader(valid_data, batch_size=32)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AMCNet1D(nn.Module):
    def __init__(self, num_channels=1, fixed_len=100):
        super(AMCNet1D, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.layer1 = ResidualBlock1D(32, 64, stride=1)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        
        self.fc1 = nn.Linear(128*10, 128)  # 512?
        self.fc2 = nn.Linear(128, fixed_len)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = AMCNet1D(num_channels=channels, fixed_len=fixed_len).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
# export train loss and validation loss
train_loss_record = np.zeros(num_epochs)
valid_loss_record = np.zeros(num_epochs)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        # evaluation
        model.eval()
        train_loss, val_loss = 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                train_loss += loss.item()
         
        # Calculate average loss over the validation set
        val_loss /= len(valid_loader)
        train_loss /= len(train_loader)
        print(f'Epoch: {epoch}, Loss: {loss.item()}', end=' ')
        print(f"Validation Loss: {val_loss}")

        train_loss_record[epoch] = train_loss
        valid_loss_record[epoch] = val_loss

np.save(f'train_history_channels_{args.channels}_{args.id}.npy', train_loss_record)
np.save(f'valid_history_channels_{args.channels}_{args.id}.npy', valid_loss_record)