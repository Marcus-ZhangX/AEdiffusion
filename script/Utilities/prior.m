function X = prior(N,d)

xmin = [3 3 10 3 9];
xmax = [5 7 13 5 11];

X = nan(N,d);

for i = 1:N
    X(i,:) = unifrnd(xmin,xmax);
end    

end