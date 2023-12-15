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
for i= 1:Ne
	[r,c]=find(obse(:,i)<0);
	obse(r,i)=obs(r);
end
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

end