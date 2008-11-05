function [s, lamcor, lamvar] = covshrink(x, shrinkvar)
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

%#author Karl Strimmer
%#modified Kevin Murphy and Matt Dunham
%#url  http://strimmerlab.org/software.html


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
x = makeMeanZero(x); x = makeStdOne(x); % convert S to R
[r, vr] = varcov(x);
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
x = makeMeanZero(x); 
S = cov(x);

if(1)       % takes only max(p*p,n*p) space
    M = zeros(p,p);
    M2 = zeros(p,p);
    N = 0;
    for i=1:n
        N = N + 1;
        newdata = kron(x(i,:)',x(i,:));
        delta = newdata - M;
        M = M + delta/N;
        M2 = M2 + delta.*(newdata-M);
    end
    VS = (M2/(n-1))* n/((n-1)^2);
else        % takes p*p*n space
  %XC1 = repmat(reshape(xc', [p 1 n]), [1 p 1]);
  %XC2 = repmat(reshape(xc', [1 p n]),  [p 1 1]); !
  %VS = var(XC1 .* XC2, 0,  3) * n/((n-1)^2);
  VS = var(bsxfun(@times,permute(x,[2,3,1]),permute(x,[3,2,1])),0,3)*n/((n-1)^2);
end


if(0)  % sanity check
    XC1test = repmat(reshape(x', [p 1 n]), [1 p 1]); % size p*p*n !
    XC2test = repmat(reshape(x', [1 p n]),  [p 1 1]); % size p*p*n !
    VStest = var(XC1test .* XC2test, 0,  3) * n/((n-1)^2);
    assert(approxeq(VS,VStest));
end

function x = makeMeanZero(x)
% make column means zero
x = bsxfun(@minus,x,mean(x));


function x = makeStdOne(x)
% make column  variances one
x = bsxfun(@rdivide,x,std(x));


