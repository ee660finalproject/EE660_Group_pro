% % % clear;
% % load fea_extr_1
% % load new_train
% % 
% % N = size (Upc_new,2);
% % type_pre = 0;
% % n=0;
% % % X = zeros(N/5,10);
% % for i = 1 : N
% %     if TripType_new(i)==type_pre
% %         col_tem = col_tem + 1;
% % %         C = [X(n,:),Upc_new(i)];
% %         [m_tem,n_tem] = size(X);
% %         if col_tem > n_tem           
% %             X = [X(1:n-1,:),zeros(n-1,1);X(n,:),Upc_new(i)];
% %         else
% %             X (n,col_tem) = Upc_new(i);
% %         end
% %     
% %     else 
% %         type_pre =  TripType_new(i);
% %         n = n+1;
% %         X(n,1) = Upc_new(i);
% %         col_tem  =1;
% %         
% %         
% %         
% %         
% %     end
% %     
% end



k = 1;
for i = 1:size (Feature_extract,1)
    if (Feature_extract(i,1)~=0)
        feature_10(k,:) = Feature_extract(i,:);
        k = k +1 ;
    end
end

fea_ind = unique(feature_10);

for i = 1:size (feature_10,1)
    for j = 1:size (feature_10,2);
        new_feature_sym(i,j) = (find(fea_ind==feature_10(i,j)));
        new_feature_num(i,j) = 10^(find(fea_ind==feature_10(i,j))-1);
    end
end

