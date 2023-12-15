clear;clc;close all
%%
Parallel_Computing = 1; 
if Parallel_Computing == 1
    Ncpu = 16;
    myCluster = parcluster('local'); % reset the maximum number of workers (i.e., cores)
    myCluster.NumWorkers = Ncpu;
    saveProfile(myCluster);
    N = maxNumCompThreads;
    LASTN = maxNumCompThreads(N);
    LASTN = maxNumCompThreads('automatic');
    parpool('local',Ncpu); % start parpool with Ncpu cores, the default profile is 'local'
end
currentdir = pwd;
addpath([currentdir,'\Utilities']);
%%
Ne = 500;
cd([currentdir,'\high_fidelity'])
copyexample(Ne);
cd(currentdir); 
%%
load Srange % range for contaminant source
ss_Ne = genex(Srange,Ne);
numColumns = size(ss_Ne, 2);
saveDirectory = "E:\Coding_path\DiffuseVAE\scripts\Surrogate_model\testing\input_testing";
for i = 1:numColumns 
    fileName = sprintf('ss%d.dat', i);
    columnData = ss_Ne(:, i); 
    fid = fopen(fullfile(saveDirectory, fileName), 'w');
    fprintf(fid, '%f\n', columnData);
    fclose(fid);
end
%%
conc_head_Ne = nan(1,Ne);%创建一个大小为 Nobs行、Ne列的数组，该数组的所有元素都被初始化为 NaN
load("E:\Coding_path\DiffuseVAE\scripts\Surrogate_model\testing\input_testing\cond_surrogate_test.mat")  % cond_200.mat是在Python中初始化时随机生成的
%%
tic
parfor i = 1:Ne
    [conc_head_Ne(:,i)] = model_H_for_surrogate(ss_Ne(:,i),cond_surrogate_test(:,i),i);
end
toc
