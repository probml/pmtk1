function suffStat = mkSuffStat(X)
% SS.n, SS.X = sum_i X(i,:)', SS.XX(j,k) = sum_i X(i,j) X(i,k)
 
n = size(X,1);
suffStat.n = n;
%suffStat.X = sum(X,1)'; % column vector
suffStat.xbar = sum(X,1)'/n; % column vector
Xc = (X-repmat(suffStat.xbar',n,1));
suffStat.XX = (Xc'*Xc)/n;
       