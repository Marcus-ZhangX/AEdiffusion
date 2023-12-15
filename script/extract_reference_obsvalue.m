clc
clear all
maindir = pwd;
addpath([maindir,'\Utilities']);%添加utilities路径
exampledir = [maindir,'\high_fidelity']; %maindir 与 \high_fidelity 相结合，创建路径
%%
t =[200:200:1000]; %5个应力期末的水头和污染物浓度
timestep = length(t);
load('obscoor.mat'); % 观测点坐标
nobs = size(obscoor,1); %nobs=观测井数
y = zeros(timestep*nobs,1); %返回5个应力期结尾的浓度
head = zeros(nobs,1);
%%
% xc = (0.75:1.5:191.25); 
% yc = (0.75:1.5:191.25);
xobs = obscoor(:,1)/1.5; 
yobs = obscoor(:,2)/1.5; 
%%
%运行正演模型
cd([exampledir,'\posterior_realization']);
system('mt3dms5b.bat');
cd(maindir);
%%
%读取浓度
CC = readMT3D([exampledir,'\posterior_realization','\MT3D001.UCN']);
for i=1:size(CC,1) 
    Ctime(i)=CC(i).time;  
end
[m,n]=find(Ctime'==t);  %获得对应应力期末尾的序号

for j = 1:timestep 
    tpcon=CC(m(j)).values;  %5个应力期，一个一个写入
    for k = 1:nobs  % nobs=25
        y(nobs*j-nobs+k,1) = tpcon(xobs(k),yobs(k)); 
    end
end
%%
%读取水头
H = readDat([exampledir,'\posterior_realization','\zx_7_12.hed']);
tphead=H.values;
for k = 1:nobs
    head(k,1) = tphead(xobs(k),yobs(k));
end
yout=[y;head];
% yout= round([y;head], 6); %将水头和浓度合并在一起作为yout  保留小数点后五位

