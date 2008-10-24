classdef gammaDist < scalarDist

  properties
    a;
    b;
  end

  %% Main methods
  methods
    function obj =  gammaDist(a,b)
      % a = shape, b = rate
      % Note that Matlab interprets b as scale=1/rate
      if nargin == 0
        a = []; b = [];
      end
      obj.a = a;
      obj.b = b;
    end

    function d = nfeatures(obj)
       d = length(obj.a);
    end
    
    function xrange = plotRange(obj) % over-ride default
      xrange = [0 3*mean(obj)];
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
      d = nfeatures(obj);
      x = X(:);
      n = length(x);
      p = zeros(n,d);
      logZ = lognormconst(obj);
      for j=1:d
        a = obj.a(j); b = obj.b(j);
        p(:,j) = (a-1) * log(x) - x.*b - logZ(j);
      end
    end

    function obj = inferParams(obj, varargin)
      % m = inferParams(model, 'name1', val1, 'name2', val2, ...)
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

  %% Demos
  methods(Static = true)
    function demoPlot(b)
      %as = [1 1.5 2  1 1.5 2]; bs = [1 1 1 1.5 1.5 1.5];
      as = [1 1.5 2];
      if nargin < 1, b = 1; end
      bs = b*ones(1,length(as));
      figure;
      [styles, colors, symbols] = plotColors;
      for i=1:length(as)
        a = as(i); b = bs(i);
        plot(gammaDist(a,b), 'xrange', [0 7], 'plotArgs', {styles{i}, 'linewidth', 2});
        hold on
        legendStr{i} = sprintf('a=%2.1f,b=%2.1f', a, b);
      end
      legend(legendStr);
      title('Gamma distributions')
    end

    function demoRainfall()
      % Demo of fitting a Gamma distribution to the rainfall data used in Rice (1995) p383
      X = dlmread('rainfallData.txt');
      X = X'; X = X(:); % concatenate across rows, not columns
      X = X(1:end-5); % removing trailing 0s
      objMoM = inferParams(gammaDist, 'data', X, 'method', 'mom');
      objMLE = inferParams(gammaDist, 'data', X, 'method', 'mle');
      [v, binc] = hist(X);
      h = binc(2)-binc(1);
      N = length(X);
      areaH = h*N;
      figure(1);clf;bar(binc, v/areaH);hold on
      xs = [0.05,  binc(end)];
      h(1)=plot(objMoM, 'xrange', xs, 'plotArgs', {'r-', 'linewidth', 3});
      h(2)=plot(objMLE, 'xrange', xs, 'plotArgs', {'k:', 'linewidth', 3});
      legend(h, 'MoM', 'MLE')
    end

  end
end