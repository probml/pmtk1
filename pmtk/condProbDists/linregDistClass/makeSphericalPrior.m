function prior = makeSphericalPrior(d, lambda, addOnes, type)
% Make a prior on w (or on (w,sigma^2) if type = 'mvnig')
% corresponding to L2 penalty, but do not penalize offset term.
% Precision = diag(0, lambda, ..., lambda) for d-1 lambdas, if addOnes=true
% Precision = diag(lambda, ..., lambda) for d lambdas, if addOnes=false
if nargin < 4, type = 'mvn'; end
%if addOnes, d=d+1; end
w0 = zeros(d,1);
precw = lambda; % prior precision
prior_precision = precw*eye(d);
if addOnes
  prior_precision(1,1) = 1e-10;
end
S0 = diag(1./diag(prior_precision)); %inv(prior_precision); % diagonal, so cheap
if strcmp(type, 'mvn')
  prior = MvnDist(w0, S0);
else
  a = 0.01; b = 0.01;
  prior = MvnInvGammaDist('mu', w0, 'Sigma', S0, 'a', a, 'b', b);
end
