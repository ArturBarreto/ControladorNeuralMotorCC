
dados = [velocidadeMotorCC_PID.Time velocidadeMotorCC_PID.Data velocidadeMotorCC_RNA.Data];

path = "C:\Users\artur\OneDrive\Documentos\MATLAB\TCC\ComparacaoPID_RNA\";
data = datetime('now','Format','yy-MM-dd_HH-mm-ss');

nomeArquivo = string(path) + 'comparacao_' + string(data) + '.csv';

cHeader = {'time' 'vel_PID' 'vel_RNA'}; %dummy header

commaHeader = [cHeader;repmat({','}, 1, numel(cHeader))];
commaHeader = commaHeader(:)';

textHeader = cell2mat(commaHeader);

fid = fopen(nomeArquivo, 'w'); 
fprintf(fid,'%s\n', textHeader);
fclose(fid);

dlmwrite(nomeArquivo, dados, '-append');

clear dados path data nomeArquivo cHeader commaHeader textHeader 
clear velocidadeMotorCC_PID velocidadeMotorCC_RNA fid
clear dadosRNA logsout tout