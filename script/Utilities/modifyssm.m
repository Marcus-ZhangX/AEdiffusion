%modify qs.ssm
%qsssm.m
function modifyssm(ii,filedir,s_lay,s_row,s_col,Ss)
delete ([filedir,'\parallel_',num2str(ii),'\zx_7_12.ssm']);
filein=[filedir,'\parallel_',num2str(ii),'\zx_7_12_as.ssm'];
fileout=[filedir,'\parallel_',num2str(ii),'\zx_7_12.ssm'];
fidin=fopen(filein,'r');
fidout=fopen(fileout,'a+');

for i=1:3
    datareadd(filein,fileout,i);
end
for i=4:63
    fprintf(fidout,'%10i%10i%10i%10.5f%10i%10.5f\r\n',s_lay,s_row(i-3),s_col,Ss(1),15,Ss(1));
end
for i=64
    fprintf(fidout,'%10i\r\n',60);
end

for i=65:124
    fprintf(fidout,'%10i%10i%10i%10.5f%10i%10.5f\r\n',s_lay,s_row(i-64),s_col,Ss(2),15,Ss(2));  
end
for i=125
    fprintf(fidout,'%10i\r\n',60);
end

for i=126:185
    fprintf(fidout,'%10i%10i%10i%10.5f%10i%10.5f\r\n',s_lay,s_row(i-125),s_col,Ss(3),15,Ss(3)); 
end

for i=186
    fprintf(fidout,'%10i\r\n',60);
end

for i=187:246
    fprintf(fidout,'%10i%10i%10i%10.5f%10i%10.5f\r\n',s_lay,s_row(i-186),s_col,Ss(4),15,Ss(4)); 
end

for i=247
    fprintf(fidout,'%10i\r\n',60);
end
for i=248:307
    fprintf(fidout,'%10i%10i%10i%10.5f%10i%10.5f\r\n',s_lay,s_row(i-247),s_col,Ss(5),15,Ss(5)); 
end

fclose(fidin);
fclose(fidout);