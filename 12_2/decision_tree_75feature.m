tic
NXtrain = X_75_train;
NYtrain = Y_75_train;
Xtest = X_75_test;
ytest =  Y_75_test;
forest = fitForest(NXtrain,NYtrain,'randomFeatures',3,'bagSize',1/3,'ntrees',10);
yhat_test = predictForest(forest,Xtest);
correct_rate = mean(yhat_test==ytest);
toc
disp('finished');