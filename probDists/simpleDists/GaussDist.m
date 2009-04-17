classdef GaussDist < ParamDist
  
  properties
    mu;
    sigma2;
  end
  
  
  %% Main methods
  methods 
     function m = GaussDist(mu, sigma2)
      % GaussDist(mu, sigma2) 
      % Note that sigma2 is the variance, not the standard deviation.
      % mu and sigma2 can be vectors; in this case, the result is a MVN with a
      % diagonal covariance matrix (product of independent 1d Gaussians).
      if nargin < 2
        sigma2 = []; mu = [];
      end
      m.mu  = mu;
      m.sigma2 = sigma2;
     end
     
     
     function d = ndistrib(obj)
       d = length(obj.mu);
     end
     
     
     function obj = mkRndParams(obj, d)
       % Set mu(j) and sigma(j) to random values, for j=1:d.
       if nargin < 2, d = ndistrib(obj); end
       obj.mu = randn(1,d);
       obj.sigma2 = rand(1,d);
     end
     
     function mu = mean(m)
       mu = m.mu;
     end
     
     function mu = mode(m)
       mu = mean(m);
     end
     
     function v = var(m)
       v = m.sigma2;
     end
     
     function [l,u] = credibleInterval(obj, p)
      if nargin < 2, p = 0.95; end
      alpha = 1-p;
      sigma = sqrt(var(obj));
      mu = obj.mu;
      l = norminv(alpha/2, mu, sigma);
      u = norminv(1-(alpha/2), mu, sigma);
      z=norminv(1-alpha/2);
      assert(approxeq(l, mu-z*sigma));
      assert(approxeq(u, mu+z*sigma));
     end
     
     function v = cdf(obj, x)
       v = normcdf(x, obj.mu, sqrt(obj.sigma2));
     end
     
     function X = sample(model, n)
       % X(i,j) = sample from gauss(m.mu(j), m.sigma(j)) for i=1:n
       if nargin < 2, n  = 1; end
       d = ndistrib(model);
       X = randn(n,d) .* repmat(sqrt(model.sigma2), n, 1) + repmat(model.mu, n, 1);
     end

     function logZ = lognormconst(obj)
       logZ = log(sqrt(2*pi*obj.sigma2));
     end
     
     function [L,Lij] = logprob(obj, X)
       % L(i) = sum_j logprob(X(i,j) | params(j))
       % Lij(i,j) = logprob(X(i,j) | params(j))
       n = size(X,1);
       d = ndistrib(obj);
       if size(X,2) == 1, X = repmat(X, 1, d); end
       logZ = lognormconst(obj);
       LZ = repmat(logZ(:)', n, 1);
       M = repmat(obj.mu(:)', n, 1);
       S2 = repmat(obj.sigma2(:)', n, 1);
       Lij = -0.5*(M-X).^2 ./ S2 - LZ;
       L = sum(Lij,2);
       %{
       LijSlow = zeros(n,d);
       for j=1:d %  
         xj = X(:,j);
         LijSlow(:,j) = (-0.5/obj.sigma2(j) .* (obj.mu(j) - xj).^2) - logZ(j);
       end
       assert(approxeq(Lij, LijSlow))
       %}
     end

     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i) = case i
      % prior - 'none' or NormInvGammDist
      % clampedMu - set to true to not update the mean
      % clampedSigma - set to true to not update the variance
      [X, prior, clampedMu, clampedSigma] = process_options(varargin, ...
        'data',[],'prior', 'none', 'clampedMu', false, 'clampedSigma',false);
      switch class(prior)
        case 'char'
          switch prior
            case 'none'
              if ~clampedMu, obj.mu = mean(X); end
              if ~clampedSigma, obj.sigma2 = var(X,1); end
            otherwise
              error(['unknown prior ' prior])
          end
        case 'NormInvGammaDist' % MAP estimation
           m = Gauss_NormInvGammaDist(prior);
           m = fit(m, 'data', X);
           post = m.muSigmaDist;
           m = mode(post);
           obj.mu = m.mu;
           obj.sigma2 = m.sigma2;
         otherwise
           error(['unknown prior '])
      end
     end
      
      function xrange = plotRange(obj, sf)
          if nargin < 2, sf = 2; end
          m = mean(obj); v = sqrt(var(obj));
          xrange = [m-sf*v, m+sf*v];
      end
      
  end
 
  
end