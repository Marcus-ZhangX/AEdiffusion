clear;clc;close all
%%
currentdir = pwd;
addpath([currentdir,'\Utilities']);
gcp;
%%
load Par.mat
Ne = Par.Ne;
cd([currentdir,'\high_fidelity'])
copyexample(Ne);
cd(currentdir);
%%
load Srange % range for contaminant source
ss_Ne = genex(Srange,Ne);
save('ss_200.mat',"ss_Ne")%将生成的"ss_Ne"的变量保存到"ss_200.mat"中（每组8个污染源）
% a = reshape(cond_200(:,1),128,128)';
%%
load('E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\z_vae_100.mat');
x1 = [z_vae_100;ss_Ne]; %z_Ne是指对应Ne数的潜在变量，ss_Ne是指对应Ne数的的污染源强参数，两者合并就是我们要反演的参数
save('x1.mat',"x1") % 保存为x1 mat文件

Nobs = 25*5+25; %观测井数量
conc_head_Ne = nan(Nobs,Ne);%创建一个大小为 Nobs行、Ne列的数组，该数组的所有元素都被初始化为 NaN
load("E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\cond_200.mat")  % cond_200.mat是在Python中初始化时随机生成的
%%
tic
parfor i = 1:Ne
    [conc_head_Ne(:,i)] = model_H(ss_Ne(:,i),cond_200(:,i),i);
end
toc

save('y1.mat',"conc_head_Ne")  % 将model_H得到的值作为观测值保存为y1

% Delete the files for parallel computation
% cd([currentdir,'\high_fidelity'])
% copyexample(Ne,-1);        %Zx 删除
% cd(currentdir);