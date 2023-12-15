function [] = ilues1()
Parallel_Computing = 0; %如果Parallel_Computing等于1，表示使用并行计算；如果等于0，表示使用串行计算
if Parallel_Computing == 1 % 如果使用并行计算
    Ncpu = 6;  % 这里的cpu数量是否可以更改？
    myCluster = parcluster('local'); % reset the maximum number of workers (i.e., cores)
    myCluster.NumWorkers = Ncpu;
    saveProfile(myCluster); % 并将之保存为一个配置文件
    N = maxNumCompThreads; % 获取当前可用的最大线程数N
    LASTN = maxNumCompThreads(N); %并将其赋值给LASTN
    LASTN = maxNumCompThreads('automatic'); % 再将LASTN设置为自动模式
    parpool('local',Ncpu); % 最后启动一个具有Ncpu个核心的本地进程池（默认配置文件为'local'）。start parpool with Ncpu cores, the default profile is 'local'
end
%%
%导入数据和参数
load Par.mat  % load the parameter settings used in ILUES(Nobs,N_Iter,Ne,alpha，range，obs，sd）
datax = load('xf.mat');%待更新参数(z_Ne和ss_Ne)
xf = datax.xf;
datay = load('yf.mat');%观测值
yf = datay.yf;


obs = Par.obs;
% obs = obs_real.*(1 + 0.05*sd.*normrnd(0,1,220,1)); 
Nobs = Par.Nobs;     % number of the measurements
Ne = Par.Ne;        % ensemble size
sd = Par.sd;
range = Par.range;
alpha = Par.alpha;
N_Iter = Par.N_Iter;
beta = sqrt(N_Iter);

%%
Cd = eye(Nobs);         %zx 生成Nobs*Nobs的对角矩阵  
for i = 1:Nobs
    Cd(i,i) = sd(i)^2;  % covariance of the measurement errors   
end

meanxf = repmat(mean(xf,2),1,Ne);           % mean of the prior parameters
Cm = (xf - meanxf)*(xf - meanxf)'/(Ne - 1); % auto-covariance of the prior parameters
      
J1 = nan(Ne,1);
for i = 1:Ne
    J1(i,1) = (yf(:,i)-obs)'/Cd*(yf(:,i)-obs);
end

xa = nan(size(xf));  % define the updated ensemble    
for j = 1:Ne
    xa(:,j) = local_update(xf,yf,Cm,sd,range,obs,alpha,beta,J1,j);
end
save xa.mat xa
	
end 

%%
%局部更新的函数
function xest = local_update(xf,yf,Cm,sd,range,obs,alpha,beta,J1,jj)

% The local updating scheme used in ILUES

Ne = size(xf,2);
xr = xf(:,jj);

xr = repmat(xr,1,Ne);
J = (xf - xr)'/Cm*(xf - xr);
J2 = diag(J);

J3 = J1/max(J1) + J2/max(J2);
M = ceil(Ne*alpha);


[J3min,index] = min(J3);
xl = xf(:,index);
yl = yf(:,index);
alfa = J3min ./ J3;
alfa(index) = 0;
index1 = RouletteWheelSelection(alfa,M-1);
xl1 = xf(:,index1);
yl1 = yf(:,index1);
xl = [xl,xl1];
yl = [yl,yl1];

xu =  updatapara(xl,yl,range,sd*beta,obs);
% xest = xu(:,randperm(M,1));
a=clock; rng(floor(sum(a(:))*10));
xest = xu(:,randperm(M,1));
% xest = xu(:,1);

end
%%
function xa = updatapara(xf,yf,range,sd,obs)

% Update the model parameters via the ensemble smoother

Npar = size(xf,1);
Ne = size(xf,2);
Nobs = length(obs);

Cd = eye(Nobs);
for i = 1:Nobs
    Cd(i,i) = sd(i)^2;
end

meanxf = repmat(mean(xf,2),1,Ne);
meanyf = repmat(mean(yf,2),1,Ne);
Cxy =  (xf - meanxf)*(yf - meanyf)'/(Ne - 1);
Cyy =  (yf - meanyf)*(yf - meanyf)'/(Ne - 1);

kgain = Cxy/(Cyy + Cd);
obse = repmat(obs,1,Ne) + normrnd(zeros(Nobs,Ne),repmat(sd,1,Ne));
xa = xf + kgain*(obse - yf);

% Boundary handling
for i = 1:Ne
    for j = 1:Npar
        if xa(j,i) > range(j,2)
            xa(j,i) = (range(j,2) + xf(j,i))/2;
        elseif xa(j,i) < range(j,1)
            xa(j,i) = (range(j,1) + xf(j,i))/2;
        end
    end
end
save("xa.mat","xa")
end
