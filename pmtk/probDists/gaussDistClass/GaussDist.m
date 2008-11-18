classdef GaussDist < ScalarDist
  
  properties
    mu;
    sigma2;
    clampedMu = false; clampedSigma = false;
  end
  
  %% Main methods
  methods 
     function m = GaussDist(mu, sigma2)
      % GaussDist(mu, sigma2) 
      % Note that sigma2 is the variance, not the standard deviation.
      % mu and sigma2 can be vectors; in this case, the result is a MVN with a
      % diagonal covariance matrix (product of independent 1d Gaussians).
      if nargin == 0
        mu = []; sigma2 = [];
      end
      m.mu  = mu;
      m.sigma2 = sigma2;
     end
     
     function d = nfeatures(obj)
       d = length(obj.mu);
     end
     
     function obj = mkRndParams(obj, d)
       % Set mu(j) and sigma(j) to random values, for j=1:d.
       if nargin < 2, d = length(obj.mu); end
       obj.mu = randn(1,d);
       obj.sigma2 = rand(1,d);
     end
     
     function mu = mean(m)
       mu = m.mu(:);
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
     
     function X = sample(m, n)
       % X(i,j) = sample from gauss(m.mu(j), m.sigma(j)) for i=1:n
       d = nfeatures(m);
       X = randn(n,d) .* repmat(sqrt(m.sigma2), n, 1) + repmat(m.mu, n, 1);
     end

     function logZ = lognormconst(obj)
       logZ = log(sqrt(2*pi*obj.sigma2));
     end
     
     function p = logprob(obj, X)
       % p(i,j) = log p(X(i) | params(j))
       d = nfeatures(obj);
       n = length(X);
       p = zeros(n,d);
       logZ = lognormconst(obj);
       for j=1:d
         p(:,j) = (-0.5/obj.sigma2(j) * (obj.mu(j) - X).^2) - logZ(j);
       end
     end

     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i) = case i
      % method - must be one of { mle, bayesian }.
      [X, suffStat, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'default');
      if any(isnan(X(:)))
        error('cannot handle missing data')
      end
      if(strcmp(method,'default'))
        if(isa(obj.mu,'double') && isa(obj.sigma2,'double'))
          method = 'mle';
        else
          method = 'bayesian';
        end
      end
      switch lower(method)
        case 'mle'
          if ~obj.clampedMu, obj.mu = mean(X); end
          if ~obj.clampedSigma, obj.sigma2 = var(X,1); end
        case 'bayesian'
          xbar = mean(X); n = size(X,1);
          switch class(obj.mu)
            case 'GaussDist'
              if ~isa(obj.sigma2, 'double'), error('Sigma must be a constant'); end
              sigma2 = obj.sigma2; tau0 = obj.mu.sigma2; mu0 = obj.mu.mu; 
              taun = (sigma2 * tau0) / (n*tau0 + sigma2);
              mun = taun*(mu0/tau0 + (n*xbar)/sigma2);
              obj.mu = GaussDist(mun, taun);
            case 'NormInvGammaDist'
              m0 = obj.mu.mu; k0 = obj.mu.k; a0 = obj.mu.a; b0 = obj.mu.b;
              kn = k0 + n;
              mn = (k0*m0 + n*xbar)/kn;
              an = a0 + n/2;
              bn = b0 + 0.5*sum((X-xbar).^2) + 0.5*n*k0*(m0-xbar)^2/(k0+n);
              obj.mu = NormInvGammaDist('mu', mn, 'k', kn, 'a', an, 'b', bn);
            case 'double'
             if ~isa(obj.sigma2, 'invGammaDist'), error('sigma must be IG'); end
             a0 = obj.sigma2.a; b0 = obj.sigma2.b;
             an = a0 +  n/2;
             bn = b0 + 0.5*sum((X-obj.mu).^2);
             obj.sigma2 = InvGammaDist(an, bn);
            otherwise
              error(['unrecognized prior ' class(obj.mu)])
          end
        otherwise
          error(['unrecognized method ' method])
      end
    end
  end
 
  
end