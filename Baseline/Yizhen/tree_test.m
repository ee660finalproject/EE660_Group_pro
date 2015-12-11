%%
load X_94247_1_5195
load X_94247_2_5195
load X_94247_3_5195
load order_for_94247
X = [X_94247_1,X_94247_2,X_94247_3];
%%
N = size(X,1);
species = cell(N,1);
for i=1:1:N
species{i,1} = num2str(Y_new(i,1));
end
%%
X = X(order(1:30000),:);
species = species(order(1:30000),:);
%%
t = classregtree(X(:,1:5195), species);
disp('finished')
%%
view(t)
%%
[cost,secost,ntermnodes,bestlevel] = test(t,'cross',X(:,1:5195),species);