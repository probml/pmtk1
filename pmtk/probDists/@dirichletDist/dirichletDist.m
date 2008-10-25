classdef dirichletDist < vecDist


  properties
    alpha;
  end

  %% Main methods
  methods
    function obj =  dirichletDist(alpha)
      if nargin == 0, alpha = []; end
      obj.alpha = alpha;
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
      X = dirichlet_sample(obj.alpha(:)',n);
    end

    function p = logprob(obj, X)
      % p(i) = log p(X(i,:) | params) where each row is a vector of size d
      % that sums to one
      p = log(X) * (obj.alpha-1) - lognormconst(obj);
    end

    function logZ = lognormconst(obj)
      a = sum(obj.alpha);
      logZ = sum(gammaln(obj.obj)) - gammaln(a);
    end

    function plot(obj) % over-ride default
      error('not supported')
    end

  end

  %% Demos
  methods(Static = true)

    function demoPlot3d()
      plotDirichlet3d([10 10 10]);
      %plotDirichlet3d([0.1 0.1 0.1]); % very slow!
    end

    function demoPlotHisto(alpha, seed)
      if nargin < 1, alpha = 0.1; end
      if nargin < 2, seed = 1; end
      rand('state', seed); randn('state', seed);
      obj = dirichletDist(alpha*ones(1,5));
      n = 5;
      probs = sample(obj, n);
      figure;
      for i=1:n
        subplot(n,1,i); bar(probs(i,:))
        if i==1, title(sprintf('Samples from Dir %3.1f', alpha)); end
      end
    end
  end

end