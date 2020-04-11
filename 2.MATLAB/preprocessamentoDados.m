% seed = 0 -> Conjunto de treino e validação
% seed = 1 -> Conjunto de testes

seed = 1;

if (seed == 0)
    quant_amostras_simulacao = length(logsout{1}.Values.Time);
    quant_steps_vel = length(logsout{2}.Values.Time);
    quant_repeticoes_step = quant_amostras_simulacao/quant_steps_vel;
    setp_simulacao = 0.02;
end

setpoint = [];

for i = 1:quant_steps_vel
    for j = 1:quant_repeticoes_step
        setpoint = [setpoint; logsout{2}.Values.Data(i)];
    end
end

if (seed == 0)
    saidaDados = [logsout{1}.Values.Time];
end

saidaDados = [saidaDados setpoint logsout{3}.Values.Data logsout{1}.Values.Data];

if(seed == 1)
    
    path = "C:\Users\artur\OneDrive\Documentos\MATLAB\TCC\EntradasRNA\";
    data = datetime('now','Format','yy-MM-dd_HH-mm-ss');
    duracaoSimulacao = quant_amostras_simulacao * setp_simulacao;
    duracaoStepVelocidade = duracaoSimulacao / quant_steps_vel;
    nomeArquivo = string(path) + 'conjuntoDadosRNA_' + string(data) + '_DuracaoSetp_' + string(duracaoStepVelocidade) + 's_DuracaoSimulacao_' + string(duracaoSimulacao) + 's.csv';
    
    %maximos = max(abs(saidaDados));
    
    %divisor_SP = max([maximos(2) maximos(5)]);
    %divisor_PV = max([maximos(4) maximos(7)]);
    
    divisores = [maximos(1) divisor_SP maximos(3) divisor_PV divisor_SP maximos(6) divisor_PV];

    for j = [2 4 5 7]
        saidaDados(:,j) =  saidaDados(:,j) / divisores(j);
    end

    cHeader = {'time' 'set point' 'output' 'process' 'set point test' 'output test' 'process test'}; %dummy header

    commaHeader = [cHeader;repmat({','}, 1, numel(cHeader))];
    commaHeader = commaHeader(:)';

    textHeader = cell2mat(commaHeader);

    fid = fopen(nomeArquivo, 'w'); 
    fprintf(fid,'%s\n', textHeader);
    fclose(fid);

    dlmwrite(nomeArquivo, saidaDados, '-append');
    
    clear ans cHeader commaHeader data duracaoSimulacao duracaoStepVelocidade fid i j logsout nomeArquivo quant_amostras_simulacao path quant_repeticoes_step quant_steps_vel saidaDados seed setp_simulacao setpoint textHeader tout
end