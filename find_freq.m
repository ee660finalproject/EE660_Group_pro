N = 642925;
unique = zeros(1,N);
unique_num = zeros(1,N);
j=1;
for i=1:1:N
    ind = find(unique==FinelineNumber_new(i))
    if isempty(ind)
        unique(j)=Upc_new(i);
        unique_num(j)=1;
        j=j+1;
    else
        unique_num(ind)=unique_num(ind)+1;
    end
end
unique_upc = unique';
unique_upc_num = unique_num';