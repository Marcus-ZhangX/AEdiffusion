function x = genex(range,N)

if nargin<2
    N = 1;
end
Npar = size(range,1);  % 计算变量 range 的行数
x = nan(Npar,N);

for i=1:N
    x(:,i) = range(:,1) + (range(:,2) - range(:,1)).*rand(Npar,1);  %在先验区间中，生成Ne个5*1的列向量
end

end

    