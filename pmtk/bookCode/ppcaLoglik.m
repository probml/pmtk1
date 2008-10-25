function ll = ppcaLoglik(X, params)
% ll(i) = log N(X(i,:) | mu, C)
% where C = W W' + sigma^2 I(d)

W = params.W; mu = params.mu; sigma2= params.sigma2;
evecs = params.evecs; evals = params.evals;

[N d] = size(X);
[d K] = size(W);
C = W*W' + sigma2*eye(d);
[p, ll] = gausspdf(X, mu, C);

if 0
% We can do this in O(K^3) time instead of O(d^3) time as follows
U = evecs(:,1:K);
L = diag(evals(1:K));
J = diag(1-sigma2/evals(1:K));
Cinv = (1/sigma2)*(eye(d) - U*J*U');
logdetC = d*log(sigma2) - sum(log(diag(J)));
MM = repmat(mu, N, 1); % replicate the mean across rows
mahal = sum(((X-MM)*Cinv).*(X-MM),2); 
logp = -0.5*mahal - (d/2)*log(2*pi) -0.5*logdetC;
assert(approxeq(ll, logp))
end
