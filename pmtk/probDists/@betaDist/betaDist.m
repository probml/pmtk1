classdef betaDist < scalarDist

  properties
    a;
    b;
  end


  %% main functions
  methods
    function obj =  betaDist(a,b)
      % betadist(a,b) propto X^{a-1} (1-X)^{b-1}
      % a and b can be vectors, in which case must be same length
      if nargin == 0, a = []; b = []; end
      obj.a = a;
      obj.b = b;
    end

    function d = nfeatures(obj)
      d = length(obj.a);
    end

    function xrange = plotRange(obj) % over-ride default
      xrange = [0 1];
    end

    function m = mean(obj)
      m = obj.a ./(obj.a + obj.b);
    end

    function m = mode(obj)
%       valid = find(obj.a + obj.b > 2);
%       d = ndims(obj);
%       m = NaN*ones(1,d);
%       m(valid) = (obj.a(valid)  - 1) ./ (obj.a(valid) + obj.b(valid) - 2);
        m = (obj.a-1)/(obj.a + obj.b -2);  
        if(isinf(m)), m = NaN; end
    end

    function m = var(obj)
      valid = find(obj.a + obj.b > 1);
      d = ndims(obj);
      m = NaN*ones(1,d);
      m(valid) = (obj.a(valid) .* obj.b(valid)) ./ ...
        ( (obj.a(valid) + obj.b(valid)).^2 .* (obj.a(valid) + obj.b(valid) + 1) );
    end


    function X = sample(obj, n)
      % X(i,j) = sample from params(j) for i=1:n
      d = ndims(obj);
      X = zeros(n, d);
      for j=1:d
        if useStatsToolbox
          X(:,j) = betarnd(obj.a(j), obj.b(j), n, 1);
        else
          error('not supported')
        end
      end
    end

    function p = logprob(obj, X, paramNdx)
      % p(i,j) = log p(x(i) | params(j))
      d = nfeatures(obj);
      if nargin < 3, paramNdx = 1:d; end
      x = X(:);
      n = length(x);
      p = zeros(n,length(paramNdx));
      for jj=1:length(paramNdx)
        j = paramNdx(jj);
        a = obj.a(j); b = obj.b(j);
        %p(:,jj) = betapdf(x, obj.a(j), obj.b(j));
        % When a==1, the density has a limit of beta(a,b) at x==0, and
        % similarly when b==1 at x==1.  Force that, instead of 0*log(0) = NaN.
        warn = warning('off','MATLAB:log:logOfZero');
        logkerna = (a-1).*log(x);   logkerna(a==1 & x==0) = 0;
        logkernb = (b-1).*log(1-x); logkernb(b==1 & x==1) = 0;
        warning(warn);
        p(:,jj) = logkerna + logkernb - betaln(a,b);
      end
    end



    function logZ = lognormconst(obj)
      logZ = betaln(obj.a, obj.b);
    end


    function obj = inferParams(obj, varargin)
      % m = inferParams(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i) = case i
      % method - mle
      [X, suffStat, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'mle');
      if useStatsToolbox
        phat = betafit(max(1e-6, min(1-1e-6,X(:)))); % prevent 0s and 1s
        %phat = betafit(X(:));
        obj.a = phat(1); obj.b = phat(2);
      else
        error('not supported')
      end
    end
  end

  %% demos
  methods(Static = true)
    function demoPlot()
      as = [0.1 1 2 8]; bs = [0.1 1 3 4];
      figure;
      [styles, colors, symbols] = plotColors;
      for i=1:length(as)
        a = as(i); b = bs(i);
        plot(betaDist(a,b), 'plotArgs', {styles{i}, 'linewidth', 2});
        hold on
        legendStr{i} = sprintf('a=%2.1f,b=%2.1f', a, b);
      end
      legend(legendStr);
      title('beta distributions')
    end

  end

end
