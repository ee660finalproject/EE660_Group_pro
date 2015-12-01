clear;
load new_train  % Read the data
N  = size(TripType_new,2); % Total input samples
j=1;
C_MAX = 44; %THE MAX of class
feature_chos = [     
%                     'UPC'
                    'DepartmentDescription'   
%                     'FinelineNumber' 
                        ];


for i = 1 : C_MAX
    if feature_chos ~='DepartmentDescription'
        fea_fre(i).feaid  = zeros(N,1);
    else
        fea_fre(i).feaid = cell(N,1);
    end
        fea_fre(i).feanum  = zeros(N,1);
end

switch feature_chos
        case 'UPC'
            Par_def=Upc_new;
        case  'DepartmentDescription' 
            Par_def=DepartmentDescription_new;
        case 'FinelineNumber' 
            Par_def=FinelineNumber_new; 
end

for i = 1:N
    
    c = TripType_new(i);
    if c == 999
       c = 1;
    end 
    
    ind = find(strcmp(fea_fre(c).feaid, Par_def(i)));
    if isempty(ind)
        fea_fre(c).feaid(j)=Par_def(i);
        fea_fre(c).feanum(j)=1;
        j=j+1;
    else
        fea_fre(c).feanum(j) = fea_fre(c).feanum(j)+1;
    end
     
%     fea_fre(c) = struct('feature',ones(10,10)) ;
%      
end


% Feature_extract = zeros(44,10);
% Feature_num = zeros(44,10);
% for c = 1:44
%     [B,I] = sort((fea_fre(c).feature(:,2)),'descend');
%     max_ind = I(1:10);
%    Feature_num(c,:) =   B(1:10)';
%     Feature_extract(c,:) = [(fea_fre(c).feature(I(1:10),1))'];
% %     pause;
% end