from sklearn.metrics.regression import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
import os
# import random
from dbn import SupervisedDBNRegression

# Parâmetros Treinamento
numCamadas = [3]
numNeuronios = [90]
# numCamadas = np.arange(1, 3, 1)
# numNeuronios = np.arange(1, 4, 1)


HIDDEN_LAYERS_STRUCTURE = []

for camadas, neuronios in product(numCamadas, numNeuronios):
    estrutura = []
    for i in range(camadas):
        estrutura.append(neuronios)
    HIDDEN_LAYERS_STRUCTURE.append(estrutura)

LEARNING_RATE_RBM = [0.01]
LEARNING_RATE = [0.01]
N_EPOCHS_RBM = [3]
N_ITER_BACKPROP = [50]
BATCH_SIZE = [16]
ACTIVATION_FUNCTION = ['sigmoid']
DROPOUT_P = [0]

indiceTreinamento = time.localtime()
indiceTreinamento = time.strftime("%Y_%m_%d_%H_%M_%S", indiceTreinamento)

indiceModelo = 0

primeiraExecucao = True

path = "./EntradasRNA/"
arquivos = sorted(os.listdir(path))

numCombinacoes = len(list(product(HIDDEN_LAYERS_STRUCTURE, LEARNING_RATE_RBM, LEARNING_RATE, N_EPOCHS_RBM,
                                  N_ITER_BACKPROP, BATCH_SIZE, ACTIVATION_FUNCTION, DROPOUT_P))) * len(arquivos)

print('\n********* Combinações que serão Testadas **********\n')
print('Total de Testes = ' + str(numCombinacoes))
print('')
print('Arquivos de teste: ' + str(arquivos))
print('HIDDEN_LAYERS_STRUCTURE = ' + str(HIDDEN_LAYERS_STRUCTURE))
print('LEARNING_RATE_RBM = ' + str(LEARNING_RATE_RBM))
print('LEARNING_RATE = ' + str(LEARNING_RATE))
print('N_EPOCHS_RBM = ' + str(N_EPOCHS_RBM))
print('N_ITER_BACKPROP = ' + str(N_ITER_BACKPROP))
print('BATCH_SIZE = ' + str(BATCH_SIZE))
print('ACTIVATION_FUNCTION = ' + str(ACTIVATION_FUNCTION))
print('DROPOUT_P = ' + str(DROPOUT_P))

for conjTreino in arquivos:
    dataset = path + conjTreino

    main_df = pd.read_csv(dataset)
    main_df.dropna(axis=1, how="all", inplace=True)

    dados = []
    datosTeste = []

    for linha in main_df.values:
        dados.append([linha[-6], linha[-4], linha[-5]])
        datosTeste.append([linha[-3], linha[-1], linha[-2]])

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

    plt.figure('Conjunto de Dados de Treinamento e Testes', figsize=(12, 7), dpi=100)

    plt.rcParams.update({'font.size': 13})

    plt.subplot(2, 2, 1)
    plt.ylim([-1.5, 2])
    plt.title('Entradas de Treinamento')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade (pu)')
    l1, l2 = plt.plot(main_df['time'], X_train)
    plt.legend((l1, l2), ('Vel. Desejada', 'Vel. Medida'), loc='upper left', frameon=False)

    plt.subplot(2, 2, 2)
    plt.title('Saídas de Treinamento')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Tensão (V)')
    plt.plot(main_df['time'], Y_train)

    plt.subplot(2, 2, 3)
    plt.ylim([-1.5, 2])
    plt.title('Entradas de Testes')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Velocidade (pu)')
    l1, l2 = plt.plot(main_df['time'], X_test)
    plt.legend((l1, l2), ('Vel. Desejada', 'Vel. Medida'), loc='upper left', frameon=False)

    plt.subplot(2, 2, 4)
    plt.title('Saídas de Testes')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Tensão (V)')
    plt.plot(main_df['time'], Y_test)

    plt.tight_layout()

    try:
        os.mkdir('./Graficos/Entradas/Treinamento_' + indiceTreinamento)
    except OSError:
        pass
    plt.savefig('./Graficos/Entradas/Treinamento_' + indiceTreinamento + '/grafico_' + os.path.splitext(conjTreino)[0])
    plt.close()

    # Treinamento
    for p_HIDDEN_LAYERS_STRUCTURE, p_LEARNING_RATE_RBM, p_LEARNING_RATE, p_N_EPOCHS_RBM, p_N_ITER_BACKPROP, \
        p_BATCH_SIZE, p_ACTIVATION_FUNCTION, p_DROPOUT_P \
            in product(HIDDEN_LAYERS_STRUCTURE, LEARNING_RATE_RBM, LEARNING_RATE, N_EPOCHS_RBM, N_ITER_BACKPROP,
                       BATCH_SIZE, ACTIVATION_FUNCTION, DROPOUT_P):

        indiceModelo = indiceModelo + 1

        print('\n***************************************************\n')
        print('Modelo ' + str(indiceModelo) + ' de ' + str(numCombinacoes))
        print('\nConjunto de treino: ' + conjTreino)
        print('Estrutura: ' + str(p_HIDDEN_LAYERS_STRUCTURE) + '\n')
        print('LEARNING_RATE_RBM = ' + str(p_LEARNING_RATE_RBM))
        print('LEARNING_RATE = ' + str(p_LEARNING_RATE))
        print('N_EPOCHS_RBM = ' + str(p_N_EPOCHS_RBM))
        print('N_ITER_BACKPROP = ' + str(p_N_ITER_BACKPROP))
        print('BATCH_SIZE = ' + str(p_BATCH_SIZE))
        print('ACTIVATION_FUNCTION = ' + str(p_ACTIVATION_FUNCTION))
        print('DROPOUT_P = ' + str(p_DROPOUT_P))

        topologia = f"{p_HIDDEN_LAYERS_STRUCTURE}"

        regressor = SupervisedDBNRegression(hidden_layers_structure=[3, 5, 2],
                                            learning_rate_rbm=p_LEARNING_RATE_RBM,
                                            learning_rate=p_LEARNING_RATE,
                                            n_epochs_rbm=p_N_EPOCHS_RBM,
                                            n_iter_backprop=p_N_ITER_BACKPROP,
                                            batch_size=p_BATCH_SIZE,
                                            activation_function=p_ACTIVATION_FUNCTION,
                                            dropout_p=p_DROPOUT_P,
                                            verbose=1)

        regressor.fit(X_train, Y_train)

        try:
            os.mkdir('./Modelos/Treinamento_' + indiceTreinamento)
        except OSError:
            pass

        regressor.save("./Modelos/Treinamento_" + indiceTreinamento + "/modelo_treinamento_" + str(indiceTreinamento) +
                       '_modelo_' + str(indiceModelo) + '_' + topologia + ".pkl")

        # Teste
        Y_pred = regressor.predict(X_test)

        r2Score = r2_score(Y_test, Y_pred)
        MSE = mean_squared_error(Y_test, Y_pred)

        print('\nDone.\nR-squared: %f\nMSE: %f' % (r2Score, MSE))

        arquivoResultados = pd.DataFrame(data={"ÍNDICE MODELO": indiceModelo,
                                               "HIDDEN_LAYERS_STRUCTURE": [p_HIDDEN_LAYERS_STRUCTURE],
                                               "r2Score": [r2Score], "MSE": [MSE],
                                               "LEARNING_RATE_RBM": [p_LEARNING_RATE_RBM],
                                               "LEARNING_RATE": [p_LEARNING_RATE], "N_EPOCHS_RBM": [p_N_EPOCHS_RBM],
                                               "N_ITER_BACKPROP": [p_N_ITER_BACKPROP], "BATCH_SIZE": [p_BATCH_SIZE],
                                               "ACTIVATION_FUNCTION": [p_ACTIVATION_FUNCTION],
                                               "DROPOUT_P": [p_DROPOUT_P], "CONJUNTO DE TREINO": conjTreino})

        arquivoResultados.to_csv(r'./Resultados/resultados_conjunto_modelos_treinamento_' + indiceTreinamento + '.csv',
                                 sep=',', index=False, mode='a', header=primeiraExecucao)

        primeiraExecucao = False

        Y_pred_train = regressor.predict(X_train)

        plt.figure('Saídas RNA', figsize=(12, 7), dpi=100)

        plt.rcParams.update({'font.size': 13})

        plt.subplot(2, 2, 1)
        plt.ylim([-1.5, 2])
        plt.title('Entradas - Dados de Treinamento')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade (pu)')
        l1, l2 = plt.plot(main_df['time'], X_train)
        plt.legend((l1, l2), ('Vel. Desejada', 'Vel. Medida'), loc='upper left', frameon=False)

        plt.subplot(2, 2, 2)
        plt.title('Saídas')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Tensão (V)')
        plt.plot(main_df['time'], Y_train)
        plt.plot(main_df['time'], Y_pred_train)
        plt.legend(('PID', 'RNA'), loc='upper left', frameon=False)

        plt.subplot(2, 2, 3)
        plt.ylim([-1.5, 2])
        plt.title('Entradas - Dados de Teste')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade (pu)')
        plt.plot(main_df['time'], X_test)
        plt.legend((l1, l2), ('Vel. Desejada', 'Vel. Medida'), loc='upper left', frameon=False)

        plt.subplot(2, 2, 4)
        plt.title('Saídas')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Tensão (V)')
        plt.plot(main_df['time'], Y_test)
        plt.plot(main_df['time'], Y_pred)
        plt.legend(('PID', 'RNA'), loc='upper left', frameon=False)

        plt.tight_layout()

        try:
            os.mkdir('./Graficos/Treinamento_' + indiceTreinamento)
        except OSError:
            pass

        plt.savefig('./Graficos/Treinamento_' + indiceTreinamento + '/grafico_treinamento_' + str(indiceTreinamento) +
                    '_modelo_' + str(indiceModelo) + '_' + topologia)
        plt.close()

        saidaRNA_treino = pd.DataFrame([main_df['time'].values, Y_pred_train]).transpose()
        saidaRNA_teste = pd.DataFrame([main_df['time'].values, Y_pred]).transpose()

        saidaRNA_treino[1] = saidaRNA_treino[1].str.get(0)
        saidaRNA_teste[1] = saidaRNA_teste[1].str.get(0)

        try:
            os.mkdir('./SaidasRNA/Treinamento_' + indiceTreinamento)
            os.mkdir('./SaidasRNA/Treinamento_' + indiceTreinamento + '/Treino')
            os.mkdir('./SaidasRNA/Treinamento_' + indiceTreinamento + '/Teste')
        except OSError:
            pass

        saidaRNA_treino.to_csv(r'./SaidasRNA/Treinamento_' + indiceTreinamento
                               + '/Treino/saida_dados_treino_treinamento_'
                               + str(indiceTreinamento) + '_modelo_' + str(indiceModelo) + '_'
                               + topologia + '.csv', index=False, header=False)
        saidaRNA_teste.to_csv(r'./SaidasRNA/Treinamento_' + indiceTreinamento +
                              '/Teste/saida_dados_teste_treinamento_' + str(indiceTreinamento) + '_modelo_' +
                              str(indiceModelo) + '_' + topologia + '.csv', index=False, header=False)
