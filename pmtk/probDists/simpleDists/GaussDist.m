classdef GaussDist < ParamDist
  
  properties
    mu;
    sigma2;
    productDist;
    prior; % for MAP estimation
  end
  
  
  %% Main methods
  methods 
     function m = GaussDist(varargin)
      % GaussDist(mu, sigma2, productDist) 
      % Note that sigma2 is the variance, not the standard deviation.
      % mu and sigma2 can be vectors
      [m.mu, m.sigma2, m.productDist, m.prior] = processArgs(varargin, ...
        '-mu', [], '-sigma2', [], '-productDist', false, '-prior', 'none');
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
       % Return col vector of log probabilities for each row of X
       % L(i) = log p(X(i) | params)
       % L(i) = log p(X(i) | params(i)) (set distrib)
       % L(i) = sum_j L(i,j)   (prod distrib)
       % L(i,j) = log p(X(i,j) | params(j))
       N = size(X,1);
       mu = obj.mu; s2 = obj.sigma2;
       logZ = lognormconst(obj);
       Lij= [];
       if ~obj.productDist
         X = X(:);
         if isscalar(mu)
           M = repmat(mu, N, 1); S2 = repmat(s2, N, 1);  LZ = repmat(logZ, N, 1);
         else
           % set distribution
           M = mu(:); S2 = s2(:);  LZ = logZ(:);
         end
         L = -0.5*(M-X).^2 ./ S2 - LZ;
       else
         if size(X,2) ~= length(mu)
           X = repmat(X(:), 1, length(mu));
           % to evaluate a set of points at a set of params
         end
         M = repmat(rowvec(mu), N, 1);
         S2 = repmat(rowvec(s2), N, 1);
         LZ = repmat(rowvec(logZ), N, 1);
         Lij = -0.5*(M-X).^2 ./ S2 - LZ;
         L = sum(Lij,2);
       end
     end
    
   
   

     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i,:) = case i. Fits vector of params, one per column.
      % prior - 'none' or NormInvGammDist
      % clampedMu - set to true to not update the mean
      % clampedSigma - set to true to not update the variance
      [X, prior, clampedMu, clampedSigma] = process_options(varargin, ...
        'data',[],'prior', obj.prior, 'clampedMu', false, 'clampedSigma',false);
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
           [obj.mu, obj.sigma2] = mode(post);
         otherwise
           error('unknown prior ')
      end
     end
      
      function xrange = plotRange(obj, sf)
          if nargin < 2, sf = 2; end
          m = mean(obj); v = sqrt(var(obj));
          xrange = [m-sf*v, m+sf*v];
      end
      
  end
 
  
end