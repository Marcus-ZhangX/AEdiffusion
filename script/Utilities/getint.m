function [dx1,dx2,n1] = getint(L,dx,x)

N = round(L/dx); % 通过将 L 除以 dx 并四舍五入得到离散化后的网格数 N
 
n1 = floor(x/(dx))+1;% 确定 x 所在网格的索引。
dx1 = x./(n1-1); % 得到一个与 x 相对应的网格间距。
n2 = N + 1 - n1;
dx2 = (L - (n1-0.5).*dx1)./(n2-0.5);

end