classdef StudentDist < ScalarDist 
  %  student T p(X|dof, mu,sigma2) 
  
  properties
    mu;
    sigma2;
    dof;
  end
  
  %% Main methods
  methods
    function m = StudentDist(dof, mu, sigma2)
      if nargin == 0
        mu = []; sigma2 = []; dof = [];
      end
      m.mu  = mu;
      m.dof = dof;
      m.sigma2 = sigma2;
    end

    function d = ndims(m)
      d = length(m.mu);
    end
    
    
    function [l,u] = credibleInterval(obj, p)
      if nargin < 2, p = 0.95; end
      alpha = 1-p;
      sigma = sqrt(var(obj));
      mu = obj.mu;
      nu = obj.dof;
      l = mu + sigma*tinv(alpha/2, nu);
      u = mu + sigma*tinv(1-(alpha/2), nu);
    end
    
    function logZ = lognormconst(obj)
      d = ndims(obj);
      v = obj.dof;
      logZ = -gammaln(v/2 + 1/2) + gammaln(v/2) + 0.5 * log(v * pi .* obj.sigma2); 
    end
    
    
    function L = logprob(obj, X)
      % L(i,j) = log p(X(i) | params(j))
      [N d] = size(X);
      d = ndims(obj);
      if d==1, X = X(:); end
      logZ = lognormconst(obj);
      for j=1:d
        v = obj.dof(j); mu = obj.mu(j); s2 = obj.sigma2(j); x = X(:,j);
        L(:,j) = (-(v+1)/2) * log(1 + (1/v)*( (x-mu).^2 / s2 ) ) - logZ(j);
      end
    end
    
   
   
     function X = sample(obj, n)
      % X(i,j) = sample ffrom params(j) i=1:n
      checkParamsAreConst(obj)
      d = ndims(obj);
      assert(statsToolboxInstalled);
      for j=1:d
        mu = repmat(obj.mu(j), n, 1);
        X(:,j) = mu + sqrt(obj.sigma2(j))*trnd(obj.dof(j), n, 1);
      end
    end

    function mu = mean(obj)
      checkParamsAreConst(obj)
      mu = obj.mu;
    end

    function mu = mode(m)
      checkParamsAreConst(m)
      mu = mean(m);
    end

    function C = var(obj)
      checkParamsAreConst(obj)
      C = (obj.dof/(obj.dof-2))*obj.sigma2;
    end
   
    function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i) = case i
      % method - currently must be mle
      [X, suffStat, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'mle');
      hasMissingData =  any(isnan(X(:)));
      if any(isnan(X(:)))
        error('cannot handle missing data')
      end
      if ~statsToolboxInstalled
        error('need stats toolbox')
      end
      if ~strcmp(lower(method), 'mle')
        error('can only handle mle')
      end
      params = mle(X, 'distribution', 'tlocationscale');
      obj.mu = params(1);
      obj.sigma2 = params(2);
      obj.dof = params(3);
    end
     
    
     
  end % methods

  methods(Static = true)
    

    function studentVsGaussianRobustnessToOutliersFigure()
      % Illustrate the robustness of the t-distribution compared to the Gaussian.
      % Written by Matthew Dunham
      gaussVsToutlierDemo;
    end
    
  end
  
  %% Private methods
  methods(Access = 'protected')
    function checkParamsAreConst(obj)
      p = isa(obj.mu, 'double') && isa(obj.sigma2, 'double') && isa(obj.dof, 'double');
      if ~p
        error('params must be constant')
      end
    end
  end
  
end