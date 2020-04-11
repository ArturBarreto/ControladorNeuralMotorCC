from sklearn.metrics.regression import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
# import random
from dbn import SupervisedDBNRegression


indiceTreinamento = time.localtime()
indiceTreinamento = time.strftime("%Y_%m_%d_%H_%M_%S", indiceTreinamento)

indiceModelo = 0

primeiraExecucao = True

path = "./EntradasRNA/"
arquivos = sorted(os.listdir(path))

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

    if conjTreino != 'degrauUnitario.csv':

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

    else:
        plt.subplot(2, 1, 1)
        plt.ylim([-1.5, 2])
        plt.title('Entradas de Testes')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade (pu)')
        l1, l2 = plt.plot(main_df['time'], X_test)
        plt.legend((l1, l2), ('Vel. Desejada', 'Vel. Medida'), loc='upper left', frameon=False)

        plt.subplot(2, 1, 2)
        plt.title('Saídas de Testes')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Tensão (V)')
        plt.plot(main_df['time'], Y_test)

    plt.tight_layout()

    try:
        os.mkdir('./Graficos/Entradas/Teste_' + indiceTreinamento)
    except OSError:
        pass
    plt.savefig('./Graficos/Entradas/Teste_' + indiceTreinamento + '/grafico_' + os.path.splitext(conjTreino)[0])
    plt.close()

    path1 = './Modelos/'

#    path2 = 'Treinamento_2019_10_11_14_40_41/'

#    path3 = 'modelo_treinamento_2019_10_11_14_40_41_modelo_101_[90, 90, 90].pkl'

    path2 = 'Treinamento_2019_11_01_14_10_48/'

    path3 = 'modelo_treinamento_2019_11_01_14_10_48_modelo_1_[90, 90, 90].pkl'

    pathCompleto = path1 + path2 + path3

    regressor = SupervisedDBNRegression.load(pathCompleto)

    # Teste
    Y_pred = regressor.predict(X_test)

    # if conjTreino == 'degrauUnitario.csv':

    #     Y_pred = Y_pred / 4.6    # 4.62073146825719

    r2Score = r2_score(Y_test, Y_pred)
    MSE = mean_squared_error(Y_test, Y_pred)

    print('\nDone.\nR-squared: %f\nMSE: %f' % (r2Score, MSE))

    arquivoResultados = pd.DataFrame(data={"Arquivo": [conjTreino], "r2Score": [r2Score], "MSE": [MSE]})

    arquivoResultados.to_csv(r'./Resultados/resultados_teste_' + indiceTreinamento + '.csv',
                             sep=',', index=False, mode='a', header=primeiraExecucao)

    primeiraExecucao = False

    Y_pred_train = regressor.predict(X_train)

    localizacao = 'upper left'
    if conjTreino == 'degrauUnitario.csv':
        # Y_pred_train = Y_pred_train / 4.6    # 4.62073146825719
        localizacao = 'lower right'

    plt.figure('Saídas RNA', figsize=(12, 7), dpi=100)

    plt.rcParams.update({'font.size': 13})

    if conjTreino != 'degrauUnitario.csv':

        plt.subplot(2, 2, 1)
        plt.ylim([-1.5, 2])
        plt.title('Entradas - Dados de Treinamento')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade (pu)')
        l1, l2 = plt.plot(main_df['time'], X_train)
        plt.legend((l1, l2), ('Vel. Desejada', 'Vel. Medida'), loc=localizacao, frameon=False)

        plt.subplot(2, 2, 2)
        plt.title('Saídas')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Tensão (V)')
        plt.plot(main_df['time'], Y_train)
        plt.plot(main_df['time'], Y_pred_train)
        plt.legend(('PID', 'RNA'), loc=localizacao, frameon=False)

        plt.subplot(2, 2, 3)
        plt.ylim([-1.5, 2])
        plt.title('Entradas - Dados de Teste')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade (pu)')
        plt.plot(main_df['time'], X_test)
        plt.legend((l1, l2), ('Vel. Desejada', 'Vel. Medida'), loc=localizacao, frameon=False)

        plt.subplot(2, 2, 4)
        plt.title('Saídas')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Tensão (V)')
        plt.plot(main_df['time'], Y_test)
        plt.plot(main_df['time'], Y_pred)
        plt.legend(('PID', 'RNA'), loc=localizacao, frameon=False)

    else:
        plt.subplot(2, 1, 1)
        plt.ylim([-1.5, 2])
        plt.title('Entradas - Dados de Teste')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Velocidade (pu)')
        plt.plot(main_df['time'], X_test)
        plt.legend((l1, l2), ('Vel. Desejada', 'Vel. Medida'), loc=localizacao, frameon=False)

        plt.subplot(2, 1, 2)
        plt.title('Saídas')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Tensão (V)')
        plt.plot(main_df['time'], Y_test)
        plt.plot(main_df['time'], Y_pred)
        plt.legend(('PID', 'RNA'), loc=localizacao, frameon=False)

    plt.tight_layout()

    try:
        os.mkdir('./Graficos/Teste_' + indiceTreinamento)
    except OSError:
        pass

    plt.savefig('./Graficos/Teste_' + indiceTreinamento + '/grafico_teste_' + str(indiceTreinamento)
                + '_' + os.path.splitext(conjTreino)[0])
    plt.close()

    saidaRNA_treino = pd.DataFrame([main_df['time'].values, Y_pred_train]).transpose()
    saidaRNA_teste = pd.DataFrame([main_df['time'].values, Y_pred]).transpose()

    saidaRNA_treino[1] = saidaRNA_treino[1].str.get(0)
    saidaRNA_teste[1] = saidaRNA_teste[1].str.get(0)

    try:
        os.mkdir('./SaidasRNA/Teste_' + indiceTreinamento)
    except OSError:
        pass

    saidaRNA_teste.to_csv(r'./SaidasRNA/Teste_' + indiceTreinamento
                          + '/saida_dados_teste_' + str(indiceTreinamento) + '.csv', index=False, header=False)
