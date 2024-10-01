import numpy as np
from matplotlib import pyplot as plt
np.random.seed(42)
from tqdm import tqdm

EPS_imag_xx = []
EPS_imag_yy = []
NATOMS_LST = []
max_files = 4000
for i in tqdm(range(1, max_files + 1)):
    data_imag = np.loadtxt('../dataset4000/imag/imag_'+str(i).zfill(4)+'.dat'\
        , skiprows=2014)
    data_real = np.loadtxt('../dataset4000/real/real_'+str(i).zfill(4)+'.dat'\
        , skiprows=2014)
    if len(data_imag) == 0:
        print('no data in', i)
        EPS_imag_xx.append(EPS_imag_xx[-1])
        EPS_imag_yy.append(EPS_imag_yy[-1])
        continue

    E_range, eps_imag_raw_xx, eps_imag_raw_yy = data_imag[:, 0], data_imag[:, 1], data_imag[:, 2]
    
    EPS_imag_xx.append(eps_imag_raw_xx[1])
    EPS_imag_yy.append(eps_imag_raw_yy[1])

EPS_yy = np.array(EPS_imag_yy)
EPS_xx = np.array(EPS_imag_xx)
fixed_len = len(EPS_xx)
#NATOMS_LST = np.array(NATOMS_LST)
# NATOMS_LST = np.ones(1000)
rdf_2d = np.load('../dataset/rdfs_2d.npy') #* NATOMS_LST[:, np.newaxis] / 200
rdf_3d = np.load('../dataset/rdfs_3d.npy') #* NATOMS_LST[:, np.newaxis] / 200
adf_raw = np.load('../dataset/adf_no_pbc.npy') # * NATOMS_LST[:, np.newaxis] / 200


if len(adf_raw[0]) < len(rdf_2d[0]):
    adf = np.zeros_like(rdf_2d)
    for i in range(len(adf_raw)):
        adf[i][:len(adf_raw[i])] = adf_raw[i]

import xgboost as xgb
from sklearn.model_selection import train_test_split


X = np.concatenate([rdf_2d, rdf_3d, adf], axis=1)
X = np.log(X + 1e-4)
# X: (N, 300)
print(X.shape)

X_train_xx, X_test_xx, y_train_xx, y_test_xx = train_test_split(X, EPS_xx, test_size=0.1, random_state=42)

model_xx = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objective function for regression (squared error)
    max_depth=4,  # Maximum depth of the trees
    learning_rate=0.1,  # Learning rate
    n_estimators=100,  # Number of trees
    eval_metric='rmse',  # Evaluation metric for regression (root mean squared error)
    # regularization
    reg_alpha=1.0,
    reg_lambda=1.0
)

X_train_yy, X_test_yy, y_train_yy, y_test_yy = train_test_split(X, EPS_yy, test_size=0.1, random_state=42)
model_yy = xgb.XGBRegressor(
    objective='reg:squarederror',  # Objective function for regression (squared error)
    max_depth=4,  # Maximum depth of the trees
    learning_rate=0.1,  # Learning rate
    n_estimators=100,  # Number of trees
    eval_metric='rmse',  # Evaluation metric for regression (root mean squared error)
    # regularization
    reg_alpha=1.0,
    reg_lambda=1.0
)

# Train the model
model_xx.fit(X_train_xx, y_train_xx)
model_yy.fit(X_train_yy, y_train_yy)

predictions = model_xx.predict(X_test_xx)
plt.figure(figsize=(4, 3))
plt.scatter(model_xx.predict(X_train_xx), y_train_xx, label='Training')
plt.scatter(predictions, y_test_xx, label='Testing')
plt.xlabel(r'Predicted $\sigma(\omega \to 0)$')
plt.ylabel(r'DFT $\sigma(\omega \to 0)$')
plt.legend()
plt.show()

predictions = model_yy.predict(X_test_yy)
plt.figure(figsize=(4, 3))
plt.scatter(model_yy.predict(X_train_yy), y_train_yy, label='Training')
plt.scatter(predictions, y_test_yy, label='Testing')
plt.xlabel(r'Predicted $\sigma(\omega \to 0)$')
plt.ylabel(r'DFT $\sigma(\omega \to 0)$')
plt.legend()
plt.show()