ind_orign = unique(VisitNumber_origin_test);
ind_new = unique(VisitNumber_new)';
N1 = size(ind_orign,1);
Y_test = zeros(N1,38);
%%
Y_prediction = double(Y_Pre');
%%
for i=1:1:N1
    i
    ind = find(ind_orign(i)==ind_new);
    if isempty(ind)
        Y_test(i,38)=1;
    else
        Y_test(i,Y_prediction(ind))=1;
    end
end
%%
xlswrite('Y_test.xls',Y_test);
disp('finished')