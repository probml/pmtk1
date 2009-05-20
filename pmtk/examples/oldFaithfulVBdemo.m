%% Reproduce figure 10.6 from 
%@book{bishop2006pra,
%  title={{Pattern recognition and machine learning}},
%  author={Bishop, C.M. and SpringerLink (Online service},
%  year={2006},
%  publisher={Springer New York.}
%}
%#author Cody Severinski
%#inprogress

setSeed(0);
load oldFaith;
% standardize the data
X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, sqrt(var(X)));
[n,d] = size(X);
K = 15;
covtype = cell(1,K);
for k=1:K
  covtype{k} = 'full';
end
alpha0 = 1e-3;
%{ old constructor
model = MixMvnVBEM(...
  '-alpha', alpha0*ones(1,K), ...
  '-mu', zeros(K,d), ...
  '-k', ones(1,K), ...
  '-T', 0.5*repmat(eye(d), [1,1,K]), ...
  '-dof', 3*ones(1,K), ...
  '-covtype', covtype);
%}
model = MixMvnVBEM('-distributions', copy(MvnInvWishartDist('mu', zeros(d,1), 'Sigma', 0.5*eye(d), 'dof', 3, 'k', 1), K, 1), '-mixingPrior', DirichletDist(alpha0*ones(K,1)));
fitted = fit(model, X, '-verbose', true, '-maxIter', 500, '-tol', 1e-10);

marg = marginalizeOutParams(fitted);

figure(); hold on; plot(X(:,1), X(:,2), 'ro');
normAlpha = normalize(fitted.mixingPrior.alpha);
for k=1:K
  % Only plot those distributions that have non-negligable contribution 
  if(normAlpha(k) > 1e-2)
    plot(marg{k});
  end
end
title(sprintf('Variational Bayesian mixture of %d Gaussians applied to Standardized old faithful dataset. \n Only mixtures with non-negligable contributions are plotted (%d).', K, sum(normAlpha > 1e-2)))