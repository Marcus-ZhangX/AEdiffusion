ka=xall(9:end,2501:end); %ȡ�����һ�θ��µ�k�ļ���
meanka=mean(ka,2);
load coor.mat  %���������Ϊgms��������
kafield=[coor meanka]; %ka������ľ�ֵ��һ�飬���淽��ȡ��k�����Ӧ������
for j=0:1:8            %ȥ����֪��hd�㣬[]�൱����ȥ����ֵ
    for i=255+250*j:5:295+j*250
        kafield(i,:)=[]
        kafield(coor(),coor())
    end
end
pp=kafield(randperm(length(kafield),10),:);   %pp=pilot point����ʣ�µ�����k���棬���ȡ10��������ֵ��