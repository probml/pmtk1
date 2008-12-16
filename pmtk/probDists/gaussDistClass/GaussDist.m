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
     
     function p = logprob(obj, X)
       % p(i,j) = log p(X(i) | params(j));
       x = X(:);
       n = length(x);
       d = ndistrib(obj);
       p = zeros(n,d);
       logZ = lognormconst(obj);
       for j=1:d % can be vectorized
         p = (-0.5/obj.sigma2(j) .* (obj.mu(j) - x).^2) - logZ(j);
       end
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
 
   %% Getters and Setters
  %{
  methods
      function obj = set.mu(obj, mu)
          obj.mu = mu;
          obj.ndims = length(mu);
      end 
  end
  %}
end