% Reproduce fig 3.9  on p65 of "Elements of statistical learning" 

load('prostate.mat') % from prostateDataMake
ndx = find(istrain);
y = y(ndx); X = X(ndx,:);
X = center(X); X = mkUnitVariance(X); y = center(y);
w = lars(X, y, 'lasso',  0, 1, [], 1);
% each row of w corresponds to a value on the reg path
wLS = X\y; denom = sum(abs(wLS'));
s = sum(abs(w),2)/denom;
figure(1);clf
plot(s,w, 'o-')
legend(names(1:8), 'location', 'northwest')
title('LASSO path on prostate cancer data')
xlabel(sprintf('shrinkage factor s(%s)', '\lambda'))
set(gca,'xlim',[0 1])
