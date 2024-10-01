import numpy as np
from matplotlib import pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

from torch.nn import functional as F

import scienceplots

#set seed
np.random.seed(0)
torch.manual_seed(0)

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

os.chdir(current_dir)

# get the changed working directory
updated_dir = os.getcwd()
print("Updated working directory:", updated_dir)

dataset = []
max_files = 1000

#input rdf
for i in range(1, max_files + 1):
    data = np.load(f'D:\本研\paper_plot\DOS/rdf_{i}.npy')
    #print(np.shape(data))
    dataset.append(data)



#input PDOS
max_freq=100
bin_size=1
bins=np.arange(0,max_freq+bin_size,bin_size)
hist=len(bins)-1
PDOS=np.zeros((max_files+1,hist))
for i in range(1, max_files + 1):
    w=np.load(f'D:\本研\paper_plot\DOS/w_{i}.npy')
    PDOS[i,:],_=np.histogram(w,bins=bins)
    # plt.plot(PDOS[i,:])
    # plt.savefig(f'result/PDOS{i}.png')
    # plt.close()

print('Done loading dataset.')

rdf=np.array(dataset)
DOSs=np.array(PDOS)[1:]
fixed_len=np.shape(DOSs)[1]   
print(np.shape(rdf),np.shape(DOSs)) 
indices=np.arange(np.shape(rdf)[0])

# Convert to PyTorch tensors
rdf = torch.tensor(rdf, dtype=torch.float32)
DOSs = torch.tensor(DOSs, dtype=torch.float32)

# Split the dataset
X_train, X_test, y_train, y_test,train_indices,test_indices= train_test_split(rdf, DOSs, indices,test_size=0.1, random_state=42)
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

np.save('train_indices.npy',train_indices)
np.save('test_indices.npy',test_indices)

# Create TensorDatasets and DataLoaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

#parameter sum: 3*64+64+64*50*50+50+50*100= 3*64+64+50*50+50+50*100= 192+64+2500+50+5000=7806
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

with plt.style.context(['science','no-latex']):
    plt.plot(loss_train)
    plt.savefig('loss.png')
    plt.close()


state_dict = model.state_dict()
numpy_params = {}

for name, param in state_dict.items():
    numpy_params[name] = param.detach().cpu().numpy()

# 将参数保存到文件
for name, param_array in numpy_params.items():
    np.save(f'{name}_params.npy', param_array)

print('Model parameters saved as NumPy arrays.')



model.eval()  # Set the model to evaluation mode
cnt = 0
result_ML=[]
result_origin=[]
ccnt=0
test_loss=[]
with torch.no_grad():
    for inputs, targets in test_loader: 
        out = model(inputs)
        target = targets.view(-1, fixed_len)  # Replace graph.y with your DOS data
        temp=target.flatten().numpy()
        temp/=np.sum(temp)
        result_origin.append(temp)
        with plt.style.context(['science','no-latex']):
            plt.figure(figsize=(6, 4.5))
            plt.plot(np.linspace(0, 100, fixed_len), target.flatten().numpy(), label='test data of DOS')
            temp=out.flatten().numpy()
            temp/=np.sum(temp)
            result_ML.append(temp)
            
            plt.plot(np.linspace(0, 100, fixed_len), out.flatten().numpy(), label='Prediction of CNN')
            plt.legend()
            plt.xlabel('Frequency/THz')
            plt.ylabel('DOS')
            plt.savefig('result/test_'+str(cnt+1).zfill(3)+'_DOS.jpg',dpi=500)
        ccnt+=1
        cnt+=1
        test_loss.append(criterion(out, target).item())

        plt.close()

result_ML=np.array(result_ML)
result_origin=np.array(result_origin)
result_ML=np.reshape(result_ML,(ccnt,-1))
result_origin=np.reshape(result_origin,(ccnt,-1))
np.save('result_ML.npy',result_ML)
np.save('result_origin.npy',result_origin)

ans=[]
for i in range(ccnt):
    peaks_origin,_=find_peaks(result_origin[i,:],distance=6,height=0.01,prominence=0.01)
    peaks_ML,_=find_peaks(result_ML[i,:],distance=6,height=0.01,prominence=0.01)
    for j in range(min(len(peaks_origin),len(peaks_ML))):
        #print(result_origin[i][p]-result_ML[i][p])
        if(peaks_ML[j]-peaks_origin[j]>2):
            continue
        ans.append((result_origin[i][peaks_origin[j]]-result_ML[i][peaks_ML[j]])/result_origin[i][peaks_origin[j]])

ans=np.array(ans)
np.save('error_peak_to_peak.npy',ans)
ans[np.where(np.abs(ans)>1)]=np.nan
with plt.style.context(['science','no-latex']):
    plt.hist(ans,bins=20)
    plt.xlim(-1,1)
    plt.savefig('error_peak_to_peak_700.svg')
    plt.close()

with plt.style.context(['science','no-latex']):
    plt.plot(test_loss)
    plt.savefig('test_loss_700.svg')
    plt.close()

