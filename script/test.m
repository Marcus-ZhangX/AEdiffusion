clc;clear all
maindir = pwd;
addpath([maindir,'\Utilities']);%添加utilities路径
exampledir = [maindir,'\high_fidelity']; %maindir 与 \high_fidelity 相结合，创建路径
%%
t = [200:200:1000]; %5个应力期末的水头和污染物浓度
timestep = length(t);
load('obscoor.mat'); % 观测点坐标
nobs = size(obscoor,1); %nobs=观测井数
y = zeros(timestep*nobs,1); %返回参数为theta时候，在观测时间点t上的浓度
head = zeros(nobs,1);%返回水头值
%%
xobs = obscoor(:,1)/1.5; %采用观测点的单位为米
yobs = obscoor(:,2)/1.5; 
%%
load('ss_200.mat')
s_row = [34:1:93];% 行  为了保持和原始文件的污染源索引顺序一致
s_col = 12; % 列
s_lay = 1;
ss = ss_Ne(:,1);
modifyssm(1,exampledir,s_lay,s_row,s_col,ss);% 修改污染源参数
%%
%修改K场
load('E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\cond_200.mat')
Tran=cond_200(:,1);
a = reshape(Tran,128,128)';
%delete([exampledir,'\parallel_',num2str(ii),'\Cond.dat']);
dlmwrite([exampledir,'\parallel_1','\Tran.dat'],Tran,'delimiter', '', 'precision', '%10.4f','newline', 'pc'); % 将新的K场写进dat文件中
%%
cd([exampledir,'\parallel_',num2str(1)]);
system('mt3dms5b.bat');
cd(maindir);
%%
%读取浓度
CC = readMT3D([exampledir,'\parallel_1','\MT3D001.UCN']);
for i=1:size(CC,1) 
    Ctime(i)=CC(i).time;  
end
[m,n]=find(Ctime'==t);  

for j = 1:timestep 
    tpcon=CC(m(j)).values;
% 	dlmwrite([outputdir,'\conc_', num2str(ii),'t_',num2str(j),'.dat'],tpcon,'delimiter',' ','precision',8) % save output(conc) of training data
    % tpcon=CC(t(j)).values;
    for k = 1:nobs    
        y(nobs*j-nobs+k,1) = tpcon(xobs(k),yobs(k)); 
    end    
end

% [X, Y] = meshgrid(1:128, 1:128);
% contour_levels = [0 10 20 50 100 150 200 300 500 700];
% contourf(X, Y, CC(100).values, contour_levels);
%%
%读取水头
H = readDat([exampledir,'\parallel_1','\zx_7_12.hed']);
tphead=H.values;
% dlmwrite([outputdir,'\head_',num2str(ii),'.dat'],tphead,'delimiter',' ','precision',8); % save output(head) of training data
for k = 1:nobs    
    head(k,1) = tphead(xobs(k),yobs(k)); 
end
yout= round([y;head], 6); %将水头和浓度合并在一起作为yout，保留小数点后五位