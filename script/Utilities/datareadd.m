function dataout=datareadd(filein,fileout,line)
fidin=fopen(filein,'r');
fidout=fopen(fileout,'a+');
nline=0;
while ~feof(fidin) % �ж��Ƿ�Ϊ�ļ�ĩβ
    tline=fgetl(fidin); % ����ָ���ļ��е���һ�У���ɾ�����з�
    nline=nline+1;
    if nline==line
        fprintf(fidout,'%s\r\n',tline);
        dataout=tline;
    end
end
fclose(fidin);
fclose(fidout);