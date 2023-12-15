function Lik = Cal_Log_Lik(y1,obs,sd)

% Log-transformed Gaussian likelihood function

[~,N] = size(y1);
Lik = zeros(N,1);

for i = 1:N
    Err = obs - y1(:,i);
    Lik(i) = - ( length(Err) / 2) * log(2 * pi) - sum ( log( sd ) ) - ...
        1/2 * sum ( ( Err./sd ).^2);
end

end