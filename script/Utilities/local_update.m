function xest = local_update(xf,yf,Cm,sd,range,obs,alpha,beta,J1,jj)

% The local updating scheme used in ILUES

Ne = size(xf,2);

J2 = nan(Ne,1);
for i = 1:Ne
    J2(i,1) = (xf(:,i) - xf(:,jj))'/Cm*(xf(:,i) - xf(:,jj));
end

J3 = J1/max(J1) + J2/max(J2);
[~,a3] = sort(J3);

M = ceil(Ne*alpha);
xl = xf(:,a3(1:M));
yl = yf(:,a3(1:M));

xu =  updatapara(xl,yl,range,sd*beta,obs);
xest = xu(:,randperm(M,1));
% xest = xu(:,1);

end