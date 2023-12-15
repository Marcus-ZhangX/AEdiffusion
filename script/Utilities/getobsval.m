function obsv = getobsval(dx1,dx2,nx1,ncx,dy1,dy2,ny1,ncy,conc,xobs,yobs)

[m,n]=size(conc);
if (m~=ncy || n ~= ncx)
    printf('error, check dim');
    pause;
end
xx = [ dx1*ones(1,nx1-1) 0.5*(dx1+dx2) dx2*ones(1,ncx-nx1-1)];
xc = [0 cumsum(xx)]; 
yy = [ dy1*ones(1,ny1-1) 0.5*(dy1+dy2) dy2*ones(1,ncy-ny1-1)];
yc = [0 cumsum(yy)]; 
obsv = interp2(xc,yc,conc,xobs,yobs,'spline');

end