function [sigma_inv f] = covselMinfunc(C,G,lambda)
% Find MLE of precision matrix given graph G
% C = cov(data), G = adjacency matrix

%#author Mark Schmidt

if nargin < 3, lambda = 0; end
nVars = length(C);

nonZero = G>0;
nonZero = setdiag(nonZero, 1);
nonZero = nonZero(:);


funObj = @(x)sparsePrecisionObjLambda(x,nVars,nonZero,C,lambda);
%K_init = diag(rand(nVars,1));
K_init = eye(nVars);
options.TolX = 1e-16;
options.TolFun = 1e-16;
%options.Method = 'newton0lbfgs';
options.Method = 'lbfgs';
options.MaxFunEvals = 1000;
options.MaxIter = 1000;
options.Display = 'off';
options.DerivativeCheck='off';
[k,f] = minFunc(funObj,K_init(nonZero),options);

K = zeros(nVars);
K(nonZero) = k;
sigma_inv = K;

end

%%%%%%%%%


function [f,g] = sparsePrecisionObjLambda(x,nVars,nonZero,sigma,lambda)

X = zeros(nVars);
X(nonZero) = x;

[R,p] = chol(X);

if p== 0
    % Fast Way to compute -logdet(X) + tr(X*sigma)
    f = -2*sum(log(diag(R))) + sum(sum(sigma.*X)) + lambda/2*sum(X(:).^2);
    g = -inv(X) + sigma + lambda*X;
    g = g(nonZero);
else
    % Matrix not in positive-definite cone, set f to Inf
    %   to force minFunc to backtrack
    f = inf;
    g = 0;
    
    % If backtracking too much:
    % optimal projection is given by projecting coefficients 
    % of spectral decomposition onto non-negative orthant)
end

end
