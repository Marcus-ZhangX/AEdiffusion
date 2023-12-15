function [] = interp_matlab()
% 这个函数是用来提取替代模型预测的6个通道（其中包含1个水头通道和5个污染物浓度应力期通道）的观测值的
load y_pred.mat
load('E:\Coding_path\DiffuseVAE\scripts\obscoor.mat'); % 观测点坐标
nobs = size(obscoor,1); %nobs=观测井数
xobs = obscoor(:,1)/1.5; %采用观测点的单位为米
yobs = obscoor(:,2)/1.5; 
y_sim = zeros(150,1); 

for i = 1:6 %y_pred是替代模型输出的预测值，一共有6个通道 ，i代表不同的通道数
	output = squeeze(y_pred(i,:,:));
    for k = 1:nobs    
       y_sim(nobs*i-nobs+k,1) = output(xobs(k),yobs(k)); 
    end
end
save y_sim.dat y_sim -ascii 

end

% a = squeeze(y_pred(6,:,:));
% a= squeeze(source(1,:,:));