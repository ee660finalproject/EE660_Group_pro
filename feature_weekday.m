% clear;

load new_train

N = size (TripType_new,2);
k = zeros(1,N);
j=1;
order = unique(Weekday_new);
%%
for i = 1 : N
    ind = find((strcmpi(order,Weekday_new{1,i})));
    k(i)=ind;
end
%%
l = 1;
X = zeros(94247,7);
for i = min(VisitNumber_new) : max(VisitNumber_new)
    ind = find(VisitNumber_new==i);
    if isempty(ind)
    else
        n = min(ind);
        X(l,k(n))=1;
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