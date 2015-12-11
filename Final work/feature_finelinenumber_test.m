% clear;
order = unique(FinelineNumber_new);
%%

N = size (FinelineNumber_new,2);
k = zeros(1,N);
j=1;

%%
for i = 1:1:N
    ind = find(order==FinelineNumber_new(1,i));
    if isempty(ind)
    k(i)= 0;
    else
    k(i)=ind;
    end
end
%%
l = 1;
X = zeros(94288,5195);
for i = min(VisitNumber_new) : max(VisitNumber_new)
    i
    ind = find(VisitNumber_new==i);
    if isempty(ind)
    else
       for j=1:1:size(ind,2)
           if k(ind(j))
              X(l,k(ind(j)))=1;                      
           end
       end
       l = l+1; 
    end
end
%%
Y_60000 = Y_new(order(1:60000),1);
X_60000 = X(order(1:60000),:);
ind_5 = find(Y_60000==5);
ind_6 = find(Y_60000==6);
ind_7 = find(Y_60000==7);
X_60000_5 = X_60000(ind_5,:);
X_60000_6 = X_60000(ind_6,:);
X_60000_7 = X_60000(ind_7,:);
%%
X_94247_1 = X(:,1:2000);
X_94247_2 = X(:,2001:4000);
X_94247_3 = X(:,4001:5195);
