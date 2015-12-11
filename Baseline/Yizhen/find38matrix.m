%%
load Y
load X_75
num = size(X_75,1);
order_type = unique(Y);
Y_new = zeros(num,1);
for i=1:1:num
    ind = find(Y(i)==order_type);
    Y_new(i)=ind;
end
%% sample set division
train_size = 60000;
num = size(X_75,1);
order = randperm(num);
X_75_train = X_75(order(1:train_size),:);
X_75_test = X_75(order(train_size+1:num),:);
Y_75_train = Y_new(order(1:train_size),:);
Y_75_test = Y_new(order(train_size+1:num),:);
%%
N1 = 30000;
N2 = N1+1;
K = 38;
Xtrain = X_75_train(1:N1,:);
Xtest = X_75_train(N2:(2*N1),:);
ytrain = Y_75_train(1:N1,:);
ytest = Y_75_train(N2:(2*N1),:);
model = knnFit(Xtrain, ytrain, K);
[yhat, yprob] = knnPredict(model, Xtest);
mean(yhat==ytest)
%%
match_matrix = zeros(38,38);
for i=1:1:size(Xtest,1)
    match_matrix(ytest(i),yhat(i))=match_matrix(ytest(i),yhat(i))+1;
end