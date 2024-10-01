import numpy as np
from matplotlib import pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from torch.nn import functional as F

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

os.chdir(current_dir)

# get the changed working directory
updated_dir = os.getcwd()
print("Updated working directory:", updated_dir)

dataset = []
max_files = 850



#input rdf
for i in range(1, max_files + 1):
    data = np.load(f'rdf/rdf_{i}.npy')
    #print(np.shape(data))
    dataset.append(data)

k=np.load('thermal.npy')[:max_files]
rdf=np.array(dataset)
DOSs=np.log10(k)
fixed_len=np.shape(DOSs)[1]   
print(np.shape(rdf),np.shape(DOSs)) 


# Convert to PyTorch tensors
rdf = torch.tensor(rdf, dtype=torch.float32)
DOSs = torch.tensor(DOSs, dtype=torch.float32)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(rdf, DOSs, test_size=0.5, random_state=42)
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create TensorDatasets and DataLoaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 49, 50)
        self.fc2 = nn.Linear(50, fixed_len)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = x.view(-1, 64 * 49)
        #x=x.view(-1,100)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN1D()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000

# Training loop
loss_train=np.zeros(num_epochs)
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        #print(inputs.shape, targets.shape)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_train[epoch]=loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

plt.plot(loss_train)
plt.savefig('loss.png')
plt.close()

model.eval()  # Set the model to evaluation mode
cnt = 0
result_ML=[]
result_origin=[]
ccnt=0
test_loss=[]

TC_predict=[]
TC_real=[]
with torch.no_grad():
    for inputs, targets in test_loader: 
        out = model(inputs)
        target = targets.view(-1, fixed_len)  # Replace graph.y with your DOS data
        TC_predict.append(out[0][0].item())
        TC_real.append(target[0][0].item())

        loss = criterion(out, target)
        test_loss.append(loss.item())

TC_train_predict=[]
TC_train_real=[]
with torch.no_grad():
    for inputs, targets in train_loader: 
        out = model(inputs)
        target = targets.view(-1, fixed_len).flatten()

        # Convert tensors to lists and append
        TC_train_predict.extend(out.view(-1).tolist())
        TC_train_real.extend(target.tolist())

np.save('TC_train_predict.npy',TC_train_predict)
np.save('TC_train_real.npy',TC_train_real)

np.save('TC_test_predict.npy',TC_predict)
np.save('TC_test_real.npy',TC_real)


TC_predict=np.array(TC_predict)
TC_real=np.array(TC_real)
TC_predict=10**TC_predict
TC_real=10**TC_real
TC_train_predict=10**np.array(TC_train_predict)
TC_train_real=10**np.array(TC_train_real)
#plot a double log graph
plt.loglog(TC_predict,TC_real,'o', markersize=0.5,c='blue')
plt.loglog(TC_train_predict,TC_train_real,'o', markersize=0.5,c='r')
plt.xlabel('predict')
plt.ylabel('real')
x=np.linspace(-1.5,1000,100)
plt.plot(x,x,'black')

plt.savefig('TC_scatter.png')
plt.close()

plt.plot(np.abs((np.array(TC_predict)-np.array(TC_real))/np.array(TC_real)))
plt.savefig('TC_error.png')
plt.close()