function w = makeSphericalPrior(m, X, lambda, type)

d = size(X,2);
w0 = zeros(d,1);
precw = lambda; % prior precision
prior_precision = precw*eye(d);
if addOffset(m.transformer)
  % We do not want to regularize the first component of the feature vector
  % (corresponding to all 1s), so we set the precision to near 0
  prior_precision(1,1) = 1e-10;
end
S0 = inv(prior_precision); % yuck!
if strcmp(type, 'mvn')
  w = mvnDist(w0, S0);
else
  a = 0.01; b = 0.01;
  w = mvnInvGammaDist('mu', w0, 'Sigma', S0, 'a', a, 'b', b);
end
