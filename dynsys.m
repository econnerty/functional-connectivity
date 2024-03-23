time_dat = ncread('sub-032304_label_ts.nc','time');
var_dat = ncread('sub-032304_label_ts.nc','__xarray_dataarray_variable__');%%% time - region - epoch
epoch_dat = ncread('sub-032304_label_ts.nc','epoch');
region_dat = ncread('sub-032304_label_ts.nc','region');
for k=1:length(epoch_dat)

%plot
for j=1:length(region_dat)%epoch_dat(end)
x(:,j)=var_dat(k,j,:);
end

x=x;
ts=0.0025;
% Inferring Interactions
for j=1:length(time_dat)-1
% delta_t(j,1) = time_dat(j+1,1)-time_dat(j,1); 
y(j,:) = [x(j+1,:)-x(j,:)]/ts;
end

 %regressor generation

phix = []; %regressor generation first order terms
for j= 1:length(x(1,:))
    phix = [phix, sin(x(1:end-1,j)) cos(x(1:end-1,j)) sin(x(1:end-1,j).*2) cos(x(1:end-1,j).*2)];
end

phix = [ones(length(x(1:end-1,1)),1), phix]; %regressor generation zeroth order terms

% fitting
    W = pinv(phix)*y;

    % self and coupling dynamic quantification
for i=1:length(x(1,:))
    [f g] = decouple(i,W);
    F(i,:)=f;
    G(:,:,i)=g;
    Gh(i,:) = sqrt(sum(G(:,:,i).^2)); 
    
    insert = @(a, x, n)cat(2,  x(1,1:n-1), a, x(1,n:end));
    if i>1
    L(i,:) = insert(0, Gh(i,:), i);
    elseif i==1
        L(i,:) = cat(2, 0, Gh(i,:));
    end
end

Coupling_strengths(:,:,k)=normalize(L);
end