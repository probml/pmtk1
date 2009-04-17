function [mu,Sigma,logZ] = laplaceApprox(logpost, start)
% Compute a Gaussian approximation to the posterior
% logpost is the unnormalized log posterior
% logpost should take a matrix as input, each row is a different param
% vector, L(i) = log p(X(i,:))

options.Method = 'cg';
options.Display = 'none';
[mu, f, g, H] = maxFuncNumerical(logpost, start, options);
ndim = length(mu); 
Sigma = -inv(H);
logZ = ndim/2*log(2*pi) + 0.5*logdet(Sigma) + f;

end


