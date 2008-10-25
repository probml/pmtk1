function [postMean, postCov] = ppcaPost(X, W, mu, sigma2)
% Probabilistic PCA - computer posterior on Z
% postMean(i,:) = E[Z|X(i,:)]
% postCov(:,:) is the same for all i

[d K] = size(W);
[N d] = size(X);
M = W'*W + sigma2*eye(K);
Minv = inv(M);
postCov = sigma2*Minv;
% use column vectors for sanity
X = X';
postMean = Minv*W'*(X-repmat(mu(:),1,N));
% convert back to row vectors
postMean = postMean';

