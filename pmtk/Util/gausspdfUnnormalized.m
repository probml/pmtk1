
function prob = gausspdfUnnormalized(X, mu, Sigma)
% prob(i) = Z * gauss(X(i,:)|mu,Sigma)
% where Z is the normalization constant
n = size(X,1);
M = repmat(mu(:)', n, 1); % replicate the mean across rows
mahal = sum(((X-M)*inv(Sigma)).*(X-M),2); % sum across features
prob = exp(-0.5*mahal); % not normalized
end
