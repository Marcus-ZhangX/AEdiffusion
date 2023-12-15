function [yout] = model_H(ss,kfield,ii)
maindir = pwd;
addpath([maindir,'\Utilities']);%添加utilities路径
exampledir = [maindir,'\high_fidelity']; %maindir 与 \high_fidelity 相结合，创建路径

% inputdir=[maindir,'\training_data\input'];%创建路径
% outputdir=[maindir,'\training_data\output'];%创建路径
% if ~exist(inputdir, 'dir')  % 如果不存在，则创建这个文件夹。
%     mkdir(inputdir);
% end
% if ~exist(outputdir, 'dir') % 如果不存在，则创建这个文件夹。
%     mkdir(outputdir);
% end
%%
t = (200:200:1000); %5个应力期末的水头和污染物浓度
timestep = length(t);
load('E:\Coding_path\DiffuseVAE\scripts\obscoor.mat'); % 观测点坐标
nobs = size(obscoor,1); %nobs=观测井数
y = zeros(timestep*nobs,1); %返回参数为theta时候，在观测时间点t上的浓度
head = zeros(nobs,1);%返回水头值
%%
%如果不使用getobsvalue的函数的话这段也就没有意义了
% xx = 17.25;
% [dx1,dx2,nx] = getint(192,1.5,xx);
% yy = [27.75,72.25,117.75,162.75];
% [dy1,dy2,ny] = getint(192,1.5,yy);
xobs = obscoor(:,1)/1.5; %采用观测点的单位为米
yobs = obscoor(:,2)/1.5; 
%%
s_row = [34:1:93];% 行  为了保持和原始文件的污染源索引顺序一致
s_col = 12; % 列
s_lay = 1;
% ss=ss_Ne(:,1)
% ii=1
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
    for k = 1:nobs    
        y(nobs*j-nobs+k,1) = tpcon(xobs(k),yobs(k)); 
    end    
end
%%
%读取水头
H = readDat([exampledir,'\parallel_',num2str(ii),'\zx_7_12.hed']);
tphead=H.values;
for k = 1:nobs    
    head(k,1) = tphead(xobs(k),yobs(k)); 
end
yout=[y;head];
% yout= round([y;head], 6); %将水头和浓度合并在一起作为yout，保留小数点后五位
end

