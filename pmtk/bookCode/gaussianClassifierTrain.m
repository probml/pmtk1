function params = gaussianClassifierTrain(X,Y)
% Input:
% X is an n x d matrix
% Y is an n-vector specifying the class label (in range 1..C)
%
% Output:
% params.mu(d,c) for feature d, class c
% params.Sigma(:,:,c) covariance for class c
% params.classPrior(c)
%
% Bayesian estimates using uninformative NG prior
% params.post.mu(d,c)
% params.post.kappa(d,c)
% params.post.alpha(d,c)
% params.post.beta(d,c)

[n d] = size(X);
Nclasses = length(unique(Y));
for c=1:Nclasses
  ndx = find(Y == c);
  dat = X(ndx, :);
  params.mu(:,c) = mean(dat);
  params.Sigma(:,:,c) = cov(dat);
  %params.sigma(:,c) = sqrt(diag(params.Sigma(:,:,c))); % diagonal
  params.sigma(:,c) = std(dat)';
  params.classPrior(c) = length(ndx)/n;
  
  m0 = 0; k0 = 0;
  %a0 = -0.5; b0 = 0;   % uninformative hyperparameters
  %a0 = 10; b0 = 10;
  a0 = 0.1; b0 = 0.1;
  nc = length(ndx); 
  for j=1:d
    xbar = mean(X(ndx,j)); s = sum( (X(ndx,j)-xbar).^2 );
    post.mu(j,c) = (k0*m0 + nc*xbar)/(k0+nc);
    post.kappa(j,c) = k0 + nc;
    post.alpha(j,c) = a0 + nc/2;
    post.beta(j,c) = b0 + 0.5*s + (k0*n*(xbar-m0)^2)/(2*(k0+nc));
    post.nc(c) = nc;
  end
end
params.post = post;
% For tied parameters
params.SigmaPooled = cov(X);
params.sigmaPooled = sqrt(diag(params.SigmaPooled));
