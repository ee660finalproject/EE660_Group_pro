tic
model = naiveBayesFit(Xtrain,ytrain,0);
disp('finish train')
ypred_val = naiveBayesPredict(model, Xval);
disp('finish val')
mean(yval==ypred_val)
toc
%%
tic
model = rvmFit(Xtrain, ytrain, 'kernelFn', @kernelLinear);
disp('finish train');
ypred_val = rvmPredict(model, Xval);
disp('finish val')
mean(yval==ypred_val)
toc
%%
tic
model = rvmFit(Xtrain, ytrain, 'kernelFn', kernelFn,'args', args)
disp('finish train')
ypred_val = rvmPredict(model, Xval);
disp('finish val')
mean(yval==ypred_val)
toc