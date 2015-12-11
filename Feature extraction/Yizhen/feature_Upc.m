order_Upc = unique(Upc_new);
type_num_Upc = zeros(38,97714);
for i=1:1:38
    i
    ind = find(Y_new==i);
    for j=1:1:size(ind,1)
        for k=1:1:209
            if(X(ind(j),k)==0)
                break;
            else
               indx = find(order_Upc ==X(ind(j),k));
               type_num_Upc(i,indx) =  type_num_Upc(i,indx)+1;
            end
        end
    end   
end
disp('finished')
%%
num_first = 23;
first_num = zeros(38,num_first);
for i=1:1:38
    i
    ind_type = sort(type_num_Upc(i,:),'descend');
    ind_location = find(type_num_Upc(i,:)>=ind_type(num_first),num_first);
    first_num(i,:)=ind_location;
end
disp('finished')
%%
ind_unique = unique(first_num);
order_first = zeros(1,size(ind_unique,1));
for i = 1:1:size(ind_unique,1)
    order_first(1,i) = order_Upc(1,ind_unique(i));
end
%%
num_in = 0;
for i=1:1:94247
    i
    for j=1:1:209
        tt=find(X(i,j)==order_first);
        if isempty(tt)
        else
            num_in = num_in+1;
            break;
        end
    end
end
disp('finished')
94247-num_in
%%
X_Upc = zeros(size(X,1),size(order_first,2));
%%
for i=1:1:size(X,1)
    i
    for j=1:1:209
        ind = find(X(i,j)==order_first);
        if isempty(ind)
        else
            X_Upc(i,ind)=1;
        end
    end
end
disp('finished')