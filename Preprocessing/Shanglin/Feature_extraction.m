% clear;
% load project_train
% tbl = tabulate(TripType);
% index = find(tbl(:,2)~=0); 
% class = tbl(index,:);
% 
% Department_ID = tabulate(DepartmentDescription);

% Fineline_ID = tabulate(FinelineNumber);

N = size(Upc,1);
% Upc_new = zeros(N,1);
% FinelineNumber_new = zeros(N,1);
k=1;
for i =1 :N
    if (isnan(Upc(i))) || (isnan(FinelineNumber(i))) ...
         || (isnan(ScanCount(i))) || (isempty(Weekday(i))) || (isempty(DepartmentDescription(i)))
        continue;
    else 
        Upc_new(k) = Upc(i);
        ScanCount_new(k) = ScanCount(i);
        FinelineNumber_new(k) = FinelineNumber(i);
        Weekday_new(k) = Weekday(i);
        DepartmentDescription_new(k) = DepartmentDescription(i);
%         TripType_new(k) =  TripType(i);
        VisitNumber_new(k) = VisitNumber(i);
        k = k+1;
    end
end


% size(unique(TripType_new))

