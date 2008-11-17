function [mu,Sigma,logZ] = laplaceApprox(start, fn, gradFn, hessFn, varargin)
% Compute a Gaussian approximation to the posterior
% fn must be of the form [f g H] = fn(params)
% where f = unnormalized posterior at params
% g = gradient
% H = Hessian

mu =  minFunc(fn,start,'method','newton');
[f g H] = fn(mu);
ndim = length(mu); 
Sigma = -inv(H);
logZ = ndim/2*log(2*pi) + 0.5*logdet(Sigma) + fn(mu);




