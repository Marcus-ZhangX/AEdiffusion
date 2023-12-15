function dataout=datareadd(filein,fileout,line)
fidin=fopen(filein,'r');
fidout=fopen(fileout,'a+');
nline=0;
while ~feof(fidin) % 判断是否为文件末尾
    tline=fgetl(fidin); % 返回指定文件中的下一行，并删除换行符
    nline=nline+1;
    if nline==line
        fprintf(fidout,'%s\r\n',tline);
        dataout=tline;
    end
end
fclose(fidin);
fclose(fidout);