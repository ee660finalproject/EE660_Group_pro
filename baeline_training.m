load Sample_data

N = size(Y,1);

index = randperm(N,1000)
X_train = X(index);
Y_train = Y(index);

