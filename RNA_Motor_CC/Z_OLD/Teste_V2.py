from sklearn.metrics.regression import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dbn import SupervisedDBNRegression

nome = "conjuntoDadosRNA_19-09-30_12-05-25_DuracaoSetp_6s_DuracaoSimulacao_60s"

dataset = "./EntradasRNA/" + nome + ".csv"

main_df = pd.read_csv(dataset)
main_df.dropna(axis=1, how="all", inplace=True)

print(main_df.head())

dados = []
datosTeste = []

for linha in main_df.values:
    dados.append([linha[-6], linha[-4], linha[-5]])
    datosTeste.append([linha[-3], linha[-1], linha[-2]])

# import random
# random.shuffle(dados)

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

plt.subplot(2, 2, 1)
plt.plot(main_df['time'], X_train)
plt.subplot(2, 2, 2)
plt.plot(main_df['time'], Y_train)

plt.subplot(2, 2, 3)
plt.plot(main_df['time'], X_test)
plt.subplot(2, 2, 4)
plt.plot(main_df['time'], Y_test)

plt.show()

# Treinamento
regressor = SupervisedDBNRegression(hidden_layers_structure=[50, 50],
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=5,
                                    n_iter_backprop=5,
                                    batch_size=16,
                                    activation_function='sigmoid',
                                    dropout_p=0.0)

regressor.fit(X_train, Y_train)

regressor.save("./Modelos/" + nome + ".pkl")

# Teste
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))

Y_pred_train = regressor.predict(X_train)

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(main_df['time'], X_train)
plt.subplot(2, 2, 2)
plt.plot(main_df['time'], Y_train)
plt.plot(main_df['time'], Y_pred_train)

plt.subplot(2, 2, 3)
plt.plot(main_df['time'], X_test)
plt.subplot(2, 2, 4)
plt.plot(main_df['time'], Y_test)
plt.plot(main_df['time'], Y_pred)
plt.show()

saidaRNA_treino = pd.DataFrame([main_df['time'].values, Y_pred_train]).transpose()
saidaRNA_teste = pd.DataFrame([main_df['time'].values, Y_pred]).transpose()

saidaRNA_treino[1] = saidaRNA_treino[1].str.get(0)
saidaRNA_teste[1] = saidaRNA_teste[1].str.get(0)

saidaRNA_treino.to_csv(r'./SaidasRNA/saida_treino_' + nome + '.csv', index=False, header=False)
saidaRNA_teste.to_csv(r'./SaidasRNA/saida_teste_' + nome + '.csv', index=False, header=False)
