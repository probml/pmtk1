classdef DirichletDist < VecDist


  properties
    alpha;
  end

  %% Main methods
  methods
    function obj =  DirichletDist(alpha)
      if nargin == 0, alpha = []; end
      obj.alpha = alpha;
    end
    
    function d = ndims(obj)
       d = numel(obj.alpha); 
    end

    function m = mean(obj)
      a = sum(obj.alpha);
      m = obj.alpha/a;
    end

    function m = mode(obj)
      a = sum(obj.alpha); k = ndims(obj);
      m = (obj.alpha-1)/(a-k);
    end

    function m = var(obj)
      % var(obj) returns a vector of marginal (component-wise) variances
      a = sum(obj.alpha);
      alpha = obj.alpha;
      m = (alpha.*(a-alpha))./(a^2*(a+1));
    end


    function X = sample(obj, n)
      % X(i,:) = random probability vector of size d that sums to one
      if(nargin < 2), n = 1;end
      X = dirichlet_sample(obj.alpha(:)',n);
    end

    function p = logprob(obj, X)
      % p(i) = log p(X(i,:) | params) where each row is a vector of size d
      % that sums to one
      p = log(X) * (obj.alpha-1) - lognormconst(obj);
    end

    function logZ = lognormconst(obj)
      a = sum(obj.alpha);
      logZ = sum(gammaln(obj.alpha)) - gammaln(a);
    end

    function plot(obj) % over-ride default
      error('not supported')
    end

  end


end