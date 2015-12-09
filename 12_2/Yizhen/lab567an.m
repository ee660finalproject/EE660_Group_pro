load x_60000_5;
load x_60000_6;
load x_60000_7;

Lable5_fea = sum(X_60000_5,1);
[ b5, ix5 ] = sort( Lable5_fea(:), 'descend' );

Lable6_fea = sum(X_60000_6,1);
[ b6, ix6 ] = sort( Lable6_fea(:), 'descend' );
Lable7_fea = sum(X_60000_7,1);
[ b7, ix7 ] = sort( Lable7_fea(:), 'descend' );

max5 = 0;
f5max = [];
i = 1;
while max5/sum(Lable5_fea)<0.9
    max5 = max5+b5(i);
    f5max = [f5max;ix5(i)];
    i = i + 1;
end


max6 = 0;
f6max = [];
i = 1;
while max6/sum(Lable6_fea)<0.9
    max6 = max6+b6(i);
    f6max = [f6max;ix6(i)];
    i = i + 1;
end 

max7 = 0;
f7max = [];
i = 1;
while max7/sum(Lable7_fea)<0.9
    max7 = max7+b7(i);
    f7max = [f7max;ix7(i)];
    i = i + 1;
end


% [max5,f5max] = max(Lable5_fea);
% [max6,f6max] = max(Lable6_fea);
% [max7,f7max] = max(Lable7_fea);

% [Lable5_fea(f5max),Lable5_fea(f6max),Lable5_fea(f7max);...
%     Lable6_fea(f5max),Lable6_fea(f6max),Lable6_fea(f7max);...
%        Lable7_fea(f5max),Lable7_fea(f6max),Lable7_fea(f7max)]
%    
% X_TRAIN_3 = [X_60000_5(:,f5max),X_60000_5(:,f6max),X_60000_5(:,f7max);...
%              X_60000_6(:,f5max),X_60000_6(:,f6max),X_60000_6(:,f7max);...
%              X_60000_7(:,f5max),X_60000_7(:,f6max),X_60000_7(:,f7max)];
%          
%          Y_TRAIN_3 = [5*ones(size(X_60000_5,1),1);...
%                       6*ones(size(X_60000_6,1),1);...
%                       7*ones(size(X_60000_7,1),1)];