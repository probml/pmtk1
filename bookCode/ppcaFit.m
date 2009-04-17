function [W,mu,sigma2,evals,evecs]  = ppcaFit(X,K)
% Probabilistic PCA - find MLEs
% Each row of X contains a feature vector, so X is n*d
% Each column of W is a pc basis
% We also return the evecs and evals of the covariance matrix
% which are useful for efficient evaluation of ppcaLoglik

[evecs, Xproj, evals, Xrecon, mu] = pcaPmtk(X,K);
[N d] = size(X);
sigma2 = mean(evals(K+1:d));
W = evecs(:,1:K) * sqrt(diag(evals(1:K))-sigma2*eye(K));
%params = struct('W', W, 'mu', mu, 'sigma2', sigma2, 'evals', evals, 'evecs', evecs);
