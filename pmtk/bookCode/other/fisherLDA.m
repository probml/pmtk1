function w = fisherLDA(X, y)
% Find the optimal vector w to separate two classes
% X is n*d
% Y(i) = 1 or 2 is the label of case i

if length(unique(y))>2
  error('can only handle binary data')
end
ndx1 = find(y==1);
ndx2 = find(y==2);
m1 = mean(X(ndx1,:))';
m2 = mean(X(ndx2,:))';
S1 = cov(X(ndx1,:));
S2 = cov(X(ndx2,:));
%SB = (m2-m1)*(m2-m1)';
SW = S1+S2;
w = inv(SW)*(m2-m1);
