%%
order_Upc = unique(Upc_new);
N = size(order_Upc,2);
ECCD = zeros(N,38);
%%
for i=1:1:size(X,1)
    i
    for j=1:1:size(X,2)
        ind=find(X(i,j)==order_Upc);
        if isempty(ind)
            break;
        else
            ECCD(ind,Y_new(i))=ECCD(ind,Y_new(i))+1;
        end      
    end
end
disp('finished')
%%
IG = zeros(N,38);
N1 = sum(sum(ECCD));
for i=1:1:size(X,1)
    i
    for j=1:1:38
        A = ECCD(i,j);
        B = sum(ECCD(i,:))-A;
        C = sum(ECCD(:,j))-A;
        D = sum(sum(ECCD))-A-B-C;
        IG(i,j) = -((A+C)/N1)*log((A+C)/N1)+(A/N1)*log(A/(A+B))+(C/N1)*log(C/(C+D));
    end
end
disp('finished')