N1 = size(IG_new,1);
max_per_row = zeros(N1,1);
for i=1:1:N1
   i
   max_per_row(i) = max(IG_new(i,:));
end
disp('finished')
%%
N4 = size(IG_new,2);
N5 = 10;
max_per_row = zeros(N1,1);
indtotal = [];
for i=1:1:N4
    i
    temp = sort(IG_new(:,i),'descend');
    for j=1:1:N1
        if  isnan(temp(j))
        else
            j = j+N5-1;
            break;
        end
    end
    indtemp = find(IG_new(:,i)>=temp(j),N5);
    indtotal = [indtotal;indtemp];
end
disp('finished')
%%
N2 = 5000;
order_de = sort(max_per_row,'descend');
ind = find(max_per_row>=order_de(N2),N2);
%%
order_upc = unique(Upc_new);
order_book = Upc_new(ind);
%%
N3 = size(X,1);
X_Upc_5000 = zeros(N3,N2);
for i=1:1:N3
  i
  for j=1:1:209
      if(X(i,j)~=0)
      ind = find(X(i,j)==order_book);
         if isempty(ind) 
         else
         X_Upc_5000(i,ind)=1;
         end
      else
         break;
      end
  end
end
disp('finished')