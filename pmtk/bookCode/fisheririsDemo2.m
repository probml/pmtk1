[X,y,classnames,varnames] = fisheririsLoad;
figure(1);clf
pscatter(X,varnames,{'ro', 'gd', 'b*'}, y)

figure(2);clf
dim = 1;
boxplot(X(:,dim), y);
set(gca,'xticklabel',classnames);
set(gca, 'xlabel', []); set(gca, 'ylabel', []);
title(sprintf('distribution of %s', varnames{dim}))

[n d] = size(X);
C = length(classnames);
XX = NaN*ones(n,C);
for c=1:C
  ndx = find(y==c);
  XX(1:length(ndx), c) = X(ndx, dim);
end
figure(3); clf
boxplot(XX)
set(gca,'xticklabel',classnames);
set('xlabel', []); set('ylabel', []);
title(sprintf('distribution of %s', varnames{dim}))


