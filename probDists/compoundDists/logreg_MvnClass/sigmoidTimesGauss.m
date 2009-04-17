function p = sigmoidTimesGauss(X, wMAP, C)
% Bishop'06 p219
mu = X*wMAP;
n = size(X,1);
if n < 1000
  sigma2 = diag(X * C * X');
else
  % to save memory, use non-vectorized version
  sigma2 = zeros(1,n);
  for i=1:n
    sigma2(i) = X(i,:)*C*X(i,:)';
  end
end
kappa = 1./sqrt(1 + pi.*sigma2./8);
p = sigmoid(kappa .* reshape(mu,size(kappa)));