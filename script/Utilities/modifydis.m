%modify Gau.dis
%Gaudis.m
function modifydis(ii,filedir,icol,irow)
delete ([filedir,'\parallel_',num2str(ii),'\Gau.dis']);
filein=[filedir,'\parallel_',num2str(ii),'\Gau_as.dis'];
fileout=[filedir,'\parallel_',num2str(ii),'\Gau.dis'];
fidin=fopen(filein,'r');
fidout=fopen(fileout,'a+');
for i=1:7
    dataread(filein,fileout,i);
end

for i=1:8
    for j=1:9
        fprintf(fidout,'%10.6f',icol(i*10+j-10));
    end
    fprintf(fidout,'%10.6f\r\n',icol(i*10));
end
% fprintf(fidout,'%10.6f\r\n',icol(81));


for i=16
    dataread(filein,fileout,i);
end

for i=1:4
    for j=1:9
        fprintf(fidout,'%10.6f',irow(i*10+j-10));
    end
    fprintf(fidout,'%10.6f\r\n',irow(i*10));
end
% fprintf(fidout,'%10.6f\r\n',irow(41));


for i=21:24
    dataread(filein,fileout,i);
end


fclose(fidin);
fclose(fidout);