%modify qs.btn
%qsbtn.m
function modifybtn(ii,filedir,icol,irow)
delete ([filedir,'\parallel_',num2str(ii),'\Gau.btn']);
filein= [filedir,'\parallel_',num2str(ii),'\Gau_as.btn'];
fileout= [filedir,'\parallel_',num2str(ii),'\Gau.btn'];
fidin=fopen(filein,'r');
fidout=fopen(fileout,'a+');
for i=1:7
    dataread(filein,fileout,i);
end

for i=1:79
    fprintf(fidout,'%10.6f',icol(i));
end
fprintf(fidout,'%10.6f\r\n',icol(80));

for i=9
    dataread(filein,fileout,i);
end

for i=1:39
    fprintf(fidout,'%10.6f',irow(i));
end
fprintf(fidout,'%10.6f\r\n',irow(40));

for i=11:42
    dataread(filein,fileout,i);
end

% for i=23
%     fprintf(fidout,'%10.6f%10i%10.0f\r\n',t1, 1, 1.0);
% end 
% for i=24
%     dataread(filein,fileout,i);
% end
% 
% for i=25
%     fprintf(fidout,'%10.6f%10i%10.0f\r\n',t2-t1, 1, 1.0);
% end 
% for i=26
%     dataread(filein,fileout,i);
% end
% 
% for i=27
%     fprintf(fidout,'%10.6f%10i%10.0f\r\n',16.0-t2, 1, 1.0);  % 16.0Ϊts
% end 
% for i=28:29
%     dataread(filein,fileout,i);
% end
fclose(fidin);
fclose(fidout);