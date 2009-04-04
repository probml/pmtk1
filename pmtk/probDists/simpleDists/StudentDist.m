classdef StudentDist < ParamDist 
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

    function model = setParams(model, param)
      model.mu = param.mu;
      model.dof = param.dof;
      model.sigma2 = param.sigma2;
    end

    function model = setParamsAlt(model, dof, mu, sigma2)
      model.dof = dof;
      model.mu = mu;
      model.sigma2 = sigma2;
    end

    function d = ndistrib(m)
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
      v = obj.dof;
      logZ = -gammaln(v/2 + 1/2) + gammaln(v/2) + 0.5 * log(v * pi .* obj.sigma2);
    end
    
    
    function [L,Lij] = logprob(obj, X)
       % L(i) = sum_j logprob(X(i,j) | params(j))
       % Lij(i,j) = logprob(X(i,j) | params(j))
      N  = size(X,1);
      d=ndistrib(obj);
      if size(X,2) == 1, X = repmat(X, 1, d); end
      %L = zeros(N,d);
      logZ = lognormconst(obj);
      v = obj.dof; mu = obj.mu; s2 = obj.sigma2;
      M = repmat(mu, N, 1);
      S2 = repmat(s2, N, 1);
      V = repmat(v, N, 1);
      LZ = repmat(logZ, N, 1);
      Lij = (-(V+1)/2) .* log(1 + (1./V).*( (X-M).^2 ./ S2 ) ) - LZ;
      %for j=1:d
      %  v = obj.dof(j); mu = obj.mu(j); s2 = obj.sigma2(j); 
      %  L(:,j) = (-(v+1)/2) * log(1 + (1/v)*( (x-mu).^2 / s2 ) ) - logZ(j);
      %end
      L = sum(Lij,2);
    end
   
   
     function X = sample(obj, n)
      % X(i,j) = sample ffrom params(j) i=1:n
      d = ndistrib(obj);
      assert(statsToolboxInstalled); %#statsToolbox
      X = zeros(n, d);
      for j=1:d
        mu = repmat(obj.mu(j), n, 1);
        X(:,j) = mu + sqrt(obj.sigma2(j))*trnd(obj.dof(j), n, 1);
      end
    end

    function mu = mean(obj)
      mu = obj.mu;
    end

    function mu = mode(m)
      mu = mean(m);
    end

    function v = var(obj)
      v = (obj.dof./(obj.dof-2))*obj.sigma2;
    end
   
    function obj = fit(obj, varargin)
        % m = fit(model, 'name1', val1, 'name2', val2, ...)
        % Finds the MLE. Needs stats toolbox.
        % Arguments are
        % data - data(i) = case i
        %
        [X, suffStat, method] = process_options(...
            varargin, 'data', [], 'suffStat', [], 'method', 'mle');
        assert(statsToolboxInstalled); %#statsToolbox
        params = mle(X, 'distribution', 'tlocationscale');
        obj.mu = params(1);
        obj.sigma2 = params(2);
        obj.dof = params(3);
    end
      
     
  end % methods

  
  
end