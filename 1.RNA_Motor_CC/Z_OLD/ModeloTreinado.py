from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from dbn.tensorflow import SupervisedDBNRegression

dataset = "./testeCSV.csv"

main_df = pd.read_csv(dataset)

print(main_df.head())

for col in main_df.columns:
    if col != 'time':
        # main_df[col] = main_df[col].pct_change()
        # main_df.dropna(inplace=True)
        # main_df[col] = preprocessing.scale(main_df[col].values)
        if col == 'set point':
            main_df[col] = (main_df[col].values / 5)
        if col == 'process':
            main_df[col] = (main_df[col].values / 4.1)
        if col == 'set point test':
            main_df[col] = (main_df[col].values / 4)
        if col == 'process test':
            main_df[col] = (main_df[col].values / 1.55)
        if col == 'output':
            main_df[col] = (main_df[col].values / 9.4)
        if col == 'output test':
            main_df[col] = (main_df[col].values / 6.1)

main_df.dropna(inplace=True)

dados = []
datosTeste = []

for linha in main_df.values:
    dados.append([linha[-6], linha[-4], linha[-5]])
    datosTeste.append([linha[-3], linha[-1], linha[-2]])

random.shuffle(dados)

# print(dados)

X_train = []
Y_train = []

X_test = []
Y_test = []

for SP, PV, OP in dados:
    X_train.append([SP, PV])
    Y_train.append(OP)

for SP, PV, OP in datosTeste:
    X_test.append([SP, PV])
    Y_test.append(OP)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

print(X_train.shape)
print(X_train)
print(Y_train.shape)
print(Y_train)

print(X_test.shape)
print(X_test)
print(Y_test.shape)
print(Y_test)

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(main_df['time'], X_train)
plt.plot(main_df['time'], Y_train)

plt.subplot(2, 1, 2)
plt.plot(main_df['time'], X_test)
plt.plot(main_df['time'], Y_test)

plt.show()

# Splitting data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
# min_max_scaler = MinMaxScaler()
# X_train = min_max_scaler.fit_transform(X_train)

regressor = SupervisedDBNRegression(hidden_layers_structure=[50, 50],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=500,
                                    n_iter_backprop=500,
                                    batch_size=16,
                                    activation_function='relu')

regressor.load("ModeloTreinado.model")

# Test
# X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))

Y_pred_train = regressor.predict(X_train)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(main_df['time'], X_train)
plt.subplot(3, 1, 2)
plt.plot(main_df['time'], Y_train)
plt.subplot(3, 1, 3)
plt.plot(main_df['time'], Y_train)
plt.plot(main_df['time'], Y_pred_train)
plt.show()

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(main_df['time'], X_test)
plt.subplot(4, 1, 2)
plt.plot(main_df['time'], Y_test)
plt.subplot(4, 1, 3)
plt.plot(main_df['time'], Y_pred)
plt.subplot(4, 1, 4)
plt.plot(main_df['time'], Y_test)
plt.plot(main_df['time'], Y_pred)
plt.show()
