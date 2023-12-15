clc
clear all
load ("E:\Coding_path\DiffuseVAE\scripts\results.mat");
num_groups = 6;
Ne=800;
num_rows_per_group = 5;
num_columns_per_group = Ne;
data = zeros(5,6*Ne);
for i = 1:5
    data(i,:) = xall(256+i,:);    %[10 1 6]
end
% 参考值
reference_values = [86.0, 187.2, 233.2, 296.7, 368.9];

% 初始化存储RMSE的数组
rmse_values = zeros(6, 1);

% 按列拆分数据以满足每个组的大小
split_data = mat2cell(data, num_rows_per_group, repmat(num_columns_per_group, 1, num_groups));

for group = 1:num_groups
    group_data = split_data{group}; % 获取当前组的数据
    num_elements = numel(group_data); % 当前组的元素数量
    s=0;
    for i = 1:5
        group_rmse = sqrt(sum((group_data(i) - reference_values(i)).^2) / num_elements); % 计算当前组的 RMSE
        s=s+group_rmse;
    end
    s=s/5;
    rmse_values(group) = s; % 存储 RMSE 值
end
