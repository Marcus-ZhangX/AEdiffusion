% 指定随机数范围和数组大小
Ne = 800;
num_singular_values = 3; % 奇异值的数量
array_size = [5, Ne]; % 5行，800列的数组大小
lower_limits = [20, 120, 210, 220, 310]; % 下限数组
upper_limits = [100, 200, 290, 300, 390]; % 上限数组

% 生成随机数数组
random_numbers = zeros(array_size);

for row = 1:5
    lower_limit = lower_limits(row);
    upper_limit = upper_limits(row);
    random_numbers(row, :) = lower_limit + (upper_limit - lower_limit) * rand(1, Ne);
end



