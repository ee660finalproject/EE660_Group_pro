%%
load('/Users/YiZheng/Desktop/EE660project/Dec.1/Y_new.mat')
load('/Users/YiZheng/Desktop/EE660project/Dec.1/X_75.mat')
load('/Users/YiZheng/Desktop/EE660project/Dec.2/X_94247_1_5195.mat')
load('/Users/YiZheng/Desktop/EE660project/Dec.2/X_94247_2_5195.mat')
load('/Users/YiZheng/Desktop/EE660project/Dec.2/X_94247_3_5195.mat')
load('/Users/YiZheng/Desktop/EE660project/Dec.5/x1.mat')
load('/Users/YiZheng/Desktop/EE660project/Dec.5/x2.mat')
load('/Users/YiZheng/Desktop/EE660project/Dec.5/x3.mat')
load('/Users/YiZheng/Desktop/EE660project/Dec.2/order_for_94247.mat')
%%
X = [X_75,X_94247_1,X_94247_2,X_94247_3,X1,X2,X3];
%%
N1 = 30000;
N2 = N1+1;
X_train = X(order(1:2*N1),:);
Y_train = Y_new(order(1:2*N1),:);
K = 38;
Xtrain = X_train(1:N1,:);
Xval = X_train(N2:(2*N1),:);
ytrain = Y_train(1:N1,:);
yval = Y_train(N2:(2*N1),:);
model = knnFit(Xtrain, ytrain, K);
[yhat, yprob] = knnPredict(model, Xval);
mean(yval~=yhat)
disp('finished')