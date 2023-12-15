function y = mfdata(x,m,n)

y = zeros(m,n);

for k = 1:m;
    for l = 1:n;
        y(k,l)=x((k-1)*n+l);
    end;
end;

end