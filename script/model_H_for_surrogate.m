function [yout] = model_H_for_surrogate(ss,kfield,ii)
%%
maindir = pwd;
addpath([maindir,'\Utilities']);%添加utilities路径
exampledir = [maindir,'\high_fidelity']; %maindir 与 \high_fidelity 相结合，创建路径

inputdir=[maindir,'\Surrogate_model\testing\input_testing'];%创建路径
outputdir=[maindir,'\Surrogate_model\testing\output_testing'];%创建路径
if ~exist(inputdir, 'dir')  % 如果不存在，则创建这个文件夹。
    mkdir(inputdir);
end
if ~exist(outputdir, 'dir') % 如果不存在，则创建这个文件夹。
    mkdir(outputdir);
end
%%
t = (200:200:1000); %5个应力期末的水头和污染物浓度
timestep = length(t);
%%
s_row = [34:1:93];% 行  为了保持和原始文件的污染源索引顺序一致
s_col = 12; % 列
s_lay = 1;
modifyssm(ii,exampledir,s_lay,s_row,s_col,ss);% 修改污染源参数
%%
%修改K场 
Tran=kfield;
%delete([exampledir,'\parallel_',num2str(ii),'\Cond.dat']);
dlmwrite([exampledir,'\parallel_',num2str(ii),'\Tran.dat'],Tran,'delimiter', '', 'precision', '%10.4f','newline', 'pc'); % 将新的K场写进dat文件中
%%
cd([exampledir,'\parallel_',num2str(ii)]);
system('mt3dms5b.bat');
cd(maindir);
%%
%读取浓度
CC = readMT3D([exampledir,'\parallel_',num2str(ii),'\MT3D001.UCN']);
for i=1:size(CC,1) 
    Ctime(i)=CC(i).time;  
end
[m,n]=find(Ctime'==t);

for j = 1:timestep 
    tpcon=CC(m(j)).values;
	dlmwrite([outputdir,'\conc_', num2str(ii),'_t',num2str(j),'.dat'],tpcon,'delimiter',' ','precision',8) % save output(conc) of training data
end
%%
%读取水头
H = readDat([exampledir,'\parallel_',num2str(ii),'\zx_7_12.hed']);
tphead=H.values;
dlmwrite([outputdir,'\head_',num2str(ii),'.dat'],tphead,'delimiter',' ','precision',8); % save output(head) of training data
yout = 1;
end

