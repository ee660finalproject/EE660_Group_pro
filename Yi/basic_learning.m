% clear;

load new_train

N = size (TripType_new,2);
k = zeros(1,N);
j=1;
order = unique(DepartmentDescription_new);
%%
for i = 1 : N
    ind = find((strcmpi(order,DepartmentDescription_new{1,i})));
    k(i)=ind;
end
%%
l = 1;
X = zeros(100000,68);
for i = min(VisitNumber_new) : max(VisitNumber_new)
    ind = find(VisitNumber_new==i);
    if isempty(ind)
    else
        for n=1:1:length(ind)
        X(l,k(ind(n)))=X(l,k(ind(n)))+ScanCount_new(ind(n));
        end
        l=l+1;
    end
end
%%
l = 1;
Y = zeros(94247,1);
for i = min(VisitNumber_new) : max(VisitNumber_new)
    ind = find(VisitNumber_new==i);
    if isempty(ind)
    else
        Y(l,1)=TripType_new(min(ind));
        l=l+1;
    end
end