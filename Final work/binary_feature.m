X = X_75(:,1:7);
%%
for i=1:1:94288
    for j=1:1:68
        if X_68(i,j)<0
            X_68(i,j)=0;
        end
        if X_68(i,j)>0
            X_68(i,j)=1;
        end
    end
end
disp('finished')
%%
X_75_train_new = [X,X_68];