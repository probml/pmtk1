classdef GammaDist < ProbDist

  properties
    a;
    b;
  end

  %% Main methods
  methods
    function obj =  GammaDist(a,b)
      % a = shape, b = rate
      % Note that Matlab interprets b as scale=1/rate
      if nargin == 0
        a = []; b = [];
      end
      obj.a = a;
      obj.b = b;
    end

    function d = ndims(obj)
       d = length(obj.a);
    end
   
    function m = mean(obj)
      m = obj.a ./ obj.b;
    end

    function m = mode(obj)
      m = (obj.a - 1) ./ obj.b;
    end

    function m = var(obj)
      m = obj.a ./ (obj.b .^ 2);
    end

    function X = sample(obj, n)
      % X(i,j) = sample from params(j) for i=1:n
      d = ndims(obj);
      X = zeros(n, d);
      for j=1:d
        if useStatsToolbox
          X(:,j) = gamrnd(obj.a(j), 1/obj.b(j), n, 1);
        else
          error('not supported')
        end
      end
    end

    function logZ = lognormconst(obj)
      logZ = gammaln(obj.a) - obj.a .* log(obj.b);
    end

    function p = logprob(obj, X)
      % p(i,j) = log p(x(i) | params(j))
      d = ndims(obj);
      x = X(:);
      n = length(x);
      p = zeros(n,d);
      logZ = lognormconst(obj);
      for j=1:d
        a = obj.a(j); b = obj.b(j);
        p(:,j) = (a-1) * log(x) - x.*b - logZ(j);
      end
    end

    function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i) = case i
      % method - one of {mle, mom} where mom  = method of moments
      [X, suffStat, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'mle');
      switch method
        case 'mle'
          if useStatsToolbox
            phat = gamfit(X);
            obj.a = phat(1); obj.b = 1/phat(2);
          else
            [a, b] = gamma_fit(X);
            obj.a = a; obj.b = 1/b; % Minka uses b=scale
          end
        case 'mom'
          xbar = mean(X); s2hat = var(X);
          obj.a = xbar^2/s2hat;
          obj.b = xbar/s2hat;
        otherwise
          error(['unknown method ' method])
      end
    end
    

  end
  
  
  methods
    
    function xrange = plotRange(obj) % over-ride default
      xrange = [0 3*mean(obj)];
    end 
      
      
  end

end