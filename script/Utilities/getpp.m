ka=xall(9:end,2501:end); %取出最后一次更新的k的集合
meanka=mean(ka,2);
load coor.mat  %这里的坐标为gms网格坐标
kafield=[coor meanka]; %ka和坐标的均值放一块，后面方便取出k及其对应的坐标
for j=0:1:8            %去掉已知的hd点，[]相当于是去掉该值
    for i=255+250*j:5:295+j*250
        kafield(i,:)=[]
        kafield(coor(),coor())
    end
end
pp=kafield(randperm(length(kafield),10),:);   %pp=pilot point，在剩下的所有k里面，随机取10个（包含值）