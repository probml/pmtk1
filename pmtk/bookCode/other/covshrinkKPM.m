function [s, lamcor, lamvar] = covshrinkKPM(x, shrinkvar)
% Shrinkage estimate of a covariance matrix, using optimal shrinkage coefficient.
% INPUT:
% x is n*p data matrix
% shrinkvar : if 1, shrinks the diagonal variance terms, default is 0
%
% OUTPUT:
% s is the posdef p*p cov matrix
% lamcor is the shrinkage coefficient for the correlaiton matrix
% lamvar is the shrinkage coefficient for the variances
%
% See  J. Schaefer and K. Strimmer.  2005.  A shrinkage approach to 
%   large-scale covariance matrix estimation and implications 
%   for functional genomics. Statist. Appl. Genet. Mol. Biol. 4:32.
% and Opgen-Rhein, R., and K. Strimmer. 2007
%   Accurate ranking of differentially expressed genes
%   by a distribution-free shrinkage approach. Statist. Appl. Genet. Mol. Biol. To appear
% This code is based on their original code http://strimmerlab.org/software.html
% but has been vectorized and simplified by Kevin Murphy.

if nargin < 2, shrinkvar = 0; end

[n p] = size(x);
if p==1, s=var(x); return; end
if shrinkvar
  [v, lamvar] = varshrink(x);
else
  v = var(x);
  lamvar = 0;
end
dsv = diag(sqrt(v));
[r, lamcor] = corshrink(x);
s = dsv*r*dsv;

%%%%%%%%

function [sv, lambda] = varshrink (x)
% Eqns 10,11 of Opgen-Rhein and Strimmer 2007
[v, vv] = varcov(x);
v = diag(v); vv = diag(vv);
vtarget = median(v);
numerator = sum(vv);
denominator = sum((v-vtarget).^2);
lambda = numerator/denominator;
lambda = min(lambda, 1); lambda = max(lambda, 0);
sv = (1-lambda)*v + lambda*vtarget;
 
function [Rhat, lambda] = corshrink(x)
% Eqns on p4 of Schafer and Strimmer 2005
[n, p] = size(x);
sx = makeMeanZero(x); sx = makeStdOne(sx); % convert S to R
[r, vr] = varcov(sx);
offdiagsumrij2 = sum(sum(tril(r,-1).^2)); 
offdiagsumvrij = sum(sum(tril(vr,-1)));
lambda = offdiagsumvrij/offdiagsumrij2;
lambda = min(lambda, 1); lambda = max(lambda, 0);
Rhat = (1-lambda)*r;
Rhat(logical(eye(p))) = 1;

function [S, VS] = varcov(x)
% s(i,j) = cov X(i,j)
% vs(i,j) = est var s(i,j)
[n,p] = size(x);
xc = makeMeanZero(x); 
S = cov(xc);
XC1 = repmat(reshape(xc', [p 1 n]), [1 p 1]); % size p*p*n !
XC2 = repmat(reshape(xc', [1 p n]),  [p 1 1]); % size p*p*n !
VS = var(XC1 .* XC2, 0,  3) * n/((n-1)^2);

function xc = makeMeanZero(x)
% make column means zero
[n,p] = size(x);
m = mean(x);
xc = x - ones(n, 1)*m; 

function xc = makeStdOne(x)
% make column  variances one
[n,p] = size(x);
sd = ones(n, 1)*std(x);
xc = x ./ sd; 


