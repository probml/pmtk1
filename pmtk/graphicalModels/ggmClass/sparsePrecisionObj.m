function [f,g] = sparsePrecisionObj(x,nVars,nonZero,sigma,lambda)

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
