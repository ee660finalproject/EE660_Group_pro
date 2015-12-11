% clear;
load new_train

N = size (TripType_new,2);
k = zeros(1,N);
j=1;
order = unique(FinelineNumber_new);
%%
for i = 1 : N
    ind = find(order==FinelineNumber_new(1,i));
    k(i)=ind;
end
%%
l = 1;
X = zeros(94247,5195);
for i = min(VisitNumber_new) : max(VisitNumber_new)
    ind = find(VisitNumber_new==i);
    if isempty(ind)
    else      
        X(l,k(ind))=1;        
        l=l+1;
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
