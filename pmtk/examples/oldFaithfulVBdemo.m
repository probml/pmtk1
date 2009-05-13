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
X = bsxfun(@rdivide, X, var(X));
[n,d] = size(X);
K = 15;
covtype = cell(1,K);
for k=1:K
  covtype{k} = 'full';
end
model = MixMvnVBEM(...
  '-alpha', 10e-3*ones(1,K), ...
  '-mu', zeros(K,d), ...
  '-k', ones(1,K), ...
  '-Sigma', 200*repmat(eye(d), [1,1,K]), ...
  '-dof', 20*ones(1,K), ...
  '-covtype', covtype);
fitted = fit(model, X, '-verbose', true, '-maxIter', 200);