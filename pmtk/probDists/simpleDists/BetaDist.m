classdef BetaDist < ProbDist

  properties
    a;
    b;
  end


  %% main functions
  methods
    function obj =  BetaDist(a,b)
      % a and b can be vectors, in which case they must be same length
      if nargin == 0, a = []; b = []; end
      obj.a = a;
      obj.b = b;
    end


    function m = mean(obj)
      m = obj.a ./(obj.a + obj.b);
      %m = [m,1-m];
    end

    function m = mode(obj)
       valid = find(obj.a + obj.b > 2);
       d = length(obj.a);
       m = NaN*ones(1,d);
       m(valid) = (obj.a(valid)  - 1) ./ (obj.a(valid) + obj.b(valid) - 2);
    end

    function m = var(obj)
      valid = find(obj.a + obj.b > 1);
      d = length(obj.a);
      m = NaN*ones(1,d);
      m(valid) = (obj.a(valid) .* obj.b(valid)) ./ ...
        ( (obj.a(valid) + obj.b(valid)).^2 .* (obj.a(valid) + obj.b(valid) + 1) );
    end


    function X = sample(obj, n)
      % X(i,j) = sample from params(j) for i=1:n
      assert(useStatsToolbox); %#statsToolbox
      d = length(obj.a);
      X = zeros(n, d);
      for j=1:d
        X(:,j) = betarnd(obj.a(j), obj.b(j), n, 1);
      end
    end

    function p = logprob(obj, X)
      % p(i,j) = log p(x(i) | params(j))
      d = length(obj.a);

      x = X(:);
      n = length(x);
      p = zeros(n,d);
      for j=1:d
        a = obj.a(j); b = obj.b(j);
        %p(:,jj) = betapdf(x, obj.a(j), obj.b(j));
        % When a==1, the density has a limit of beta(a,b) at x==0, and
        % similarly when b==1 at x==1.  Force that, instead of 0*log(0) = NaN.
        warn = warning('off','MATLAB:log:logOfZero');
        logkerna = (a-1).*log(x);   logkerna(a==1 & x==0) = 0;
        logkernb = (b-1).*log(1-x); logkernb(b==1 & x==1) = 0;
        warning(warn);
        p(:,j) = logkerna + logkernb - betaln(a,b);
      end
    end
         


    function logZ = lognormconst(obj)
      logZ = betaln(obj.a, obj.b);
    end


    function obj = fit(obj, varargin)
      % m = fit(model, data)
      % Arguments are
      % data - data(i) = case i
      % Compute MLE. Needs stats toolbox.
      [X] = processArgs(varargin, '-data', []);
      assert(useStatsToolbox); %#statsToolbox
      phat = betafit(max(1e-6, min(1-1e-6,X(:)))); % prevent 0s and 1s
      %phat = betafit(X(:));
      obj.a = phat(1); obj.b = phat(2);
    end
    
    function h=plot(obj, varargin)
      [plotArgs, npoints] = processArgs(...
        varargin, '-plotArgs' ,{}, '-npoints', 100);
     xs = linspace(0, 1, npoints); 
     p = exp(logprob(obj, xs(:)));
     h = plot(colvec(xs), colvec(p), plotArgs{:});
    end
                
  end
    

end
