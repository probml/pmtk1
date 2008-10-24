classdef invGammaDist < scalarDist

  properties
    a;
    b;
  end

  %% Main methods
  methods
    function obj =  invGammaDist(a,b)
      % a = shape, b = scale
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
      m = obj.b ./ (obj.a-1);
    end

    function m = mode(obj)
      m = obj.b ./ (obj.a + 1);
    end

    function m = var(obj)
      m = (obj.b.^2) ./ ( (obj.a-1).^2 .* (obj.a-2) );
    end


    function X = sample(obj, n)
      % X(i,j) = sample from params(j) for i=1:n
      d = nfeatures(obj);
      X = zeros(n, d);
      for j=1:d
        v = 2*obj.a(j);
        s2 = 2*obj.b(j)/v;
        X(:,j) = invchi2rnd(v, s2, n, 1);
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
        p(:,j) = -(a+1) * log(x) - b./x - logZ(j);
      end
    end

  end

  %% Demos
  methods(Static = true)
     
    function demoPlot(small)
      if nargin < 1, small = false; end
      if small
        as = [0.01 0.1 1];
        bs = as;
        xr = [0 2];
      else
        as = [0.1 0.5 1 2];
        bs = 1*ones(1,length(as));
        xr = [0 5];
      end
      figure;
      [styles, colors, symbols] = plotColors;
      for i=1:length(as)
        a = as(i); b = bs(i);
        plot(invGammaDist(a,b), 'xrange', xr, 'plotArgs', {styles{i}, 'linewidth', 2});
        hold on
        legendStr{i} = sprintf('a=%4.3f,b=%4.3f', a, b);
      end
      legend(legendStr);
      title('InvGamma distributions')
    end
    
    function demoSample(small)
      setSeed(0);
      if nargin < 1, small = false; end
      if small
        as = [0.01 0.1 1]; bs = as;
      else
        as = [0.1 0.5 1 2];
        bs = 1*ones(1,length(as));
      end
      figure;
      for i=1:length(as)
        a = as(i); b = bs(i);
        XX = sample(invGammaDist(a,b), 1000);
        subplot(length(as),1,i);
        hist(XX);
        title(sprintf('a=%4.3f,b=%4.3f', a, b))
      end
    end
  end

end