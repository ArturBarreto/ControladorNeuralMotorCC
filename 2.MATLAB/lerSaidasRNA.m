% opcao = 1 -> Saída teste do Modelo escolhido
% opcao = 2 -> Saída da função degrau

opcao = 1;

path1 = "C:\Users\artur\OneDrive\Documentos\0. Ubuntu\DBN\SaidasRNA\";

if (opcao == 1)
    path2 = "Treinamento_2019_10_11_14_40_41\";

    path3 = "Teste\";

    path4 = "saida_dados_teste_treinamento_2019_10_11_14_40_41_modelo_101_[90, 90, 90].csv";
else 
    path2 = "Teste_2019_10_31_13_10_35\";

    path3 = "saida_dados_teste_2019_10_31_13_10_35";

    path4 = ".csv";
end
    
pathCompleto = path1 + path2 + path3 + path4;

dadosRNA = csvread(pathCompleto);

clear path1 path2 path3 path4 pathCompleto opcao