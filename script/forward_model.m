function forward_model()
distcomp.feature( 'LocalUseMpiexec', false );  % 防止并行报错
Parallel_Computing = 1; % 0 for serial computing, 1 for parallel computing
if Parallel_Computing == 1
    Ncpu = 6;
    myCluster = parcluster('local'); % reset the maximum number of workers (i.e., cores)
    myCluster.NumWorkers = Ncpu;
    saveProfile(myCluster);
    N = maxNumCompThreads;
    LASTN = maxNumCompThreads(N);
    LASTN = maxNumCompThreads('automatic');
    parpool('local',Ncpu); % start parpool with Ncpu cores, the default profile is 'local'
end

load Par.mat
Ne = Par.Ne;
Nobs = Par.Nobs;
load ('E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\z_a.mat')
load ('E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\ss_a.mat')
load ('E:\Coding_path\DiffuseVAE\scripts\ILUES_vars\cond_a.mat')
a = ss_a;
b = cond_a;

ya = nan(Nobs,Ne);

parfor i = 1:Ne
    [ya(:,i)] = model_H(a(:,i),b(:,i),i);
end

save('ya.mat', "ya")

delete(gcp('nocreate'));
end