classdef GammaDist < ProbDist

  properties
    a;
    b;
    fitMethod;
  end

  %% Main methods
  methods
    function obj =  GammaDist(varargin)
      % GammaDist(a = shape, b = rate, fitMethod)
      % Note that Matlab interprets b as scale=1/rate
      % fitMethod is 'mle' or 'mom' (method of moments)
      [obj.a, obj.b, obj.fitMethod] = processArgs(...
        varargin, '-a', [], '-b', [], '-fitMethod', 'mle');
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
      d = length(obj.a);
      X = zeros(n, d);
      for j=1:d
        if useStatsToolbox %#statsToolbox
          X(:,j) = gamrnd(obj.a(j), 1/obj.b(j), n, 1);
        else
          error('not supported')
        end
      end
    end

    function logZ = lognormconst(obj)
      logZ = gammaln(obj.a) - obj.a .* log(obj.b);
    end

    function [L,Lij] = logprob(obj, X)
      % Return col vector of log probabilities for each row of X
      % If X(i) is a scalar:
      % L(i) = log p(X(i,1) | params)
      % L(i) = log p(X(i,1) | params(i))
      % If X(i,:) is a vector:
      % Lij(i,j) = log p(X(i,j) | params(j))
      % L(i) = sum_j Lij(i,j)   (product distrib)
      [nx,nd] = size(X);
      aa = obj.a; bb = obj.b;
      d = length(aa);
      logZ = lognormconst(obj);
      if nd==1 % scalar data
        if d==1 % replicate parameter
          A = repmat(aa, nx, 1); B = repmat(bb, nx, 1); LZ = repmat(logZ, nx, 1);
        else % one param per case
          A = aa(:); B = bb(:);  LZ = logZ(:);
          assert(length(A)==nx);
        end
      else % vector data
        %assert(obj.productDist);
        assert(nd==d);
        A = repmat(rowvec(aa), nx, 1);
        B = repmat(rowvec(bb), nx, 1);
        LZ = repmat(rowvec(logZ), nx, 1);
      end
      %Lij(:,j) = (a(j)-1) * log(X(:,j)) - X(:,j).*b(j) - logZ(j);
      Lij = (A-1) .* log(X) - X.*B - LZ;
      L = sum(Lij,2);
    end

    function obj = fit(obj, varargin)
      % m = fit(model, data, SS)
      [X, suffStat] = processArgs(varargin, '-data', [], '-SS', []);
      assert(~isempty(X))
      switch obj.fitMethod
        case 'mle'
          if useStatsToolbox %#statsToolbox
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
          error(['unknown method ' obj.fitMethod])
      end
    end
    
    function h=plot(obj, varargin)
      [plotArgs, npoints, xrange] = processArgs(...
        varargin, '-plotArgs' ,{}, '-npoints', 100, ...
        '-xrange', [0 3*mean(obj)]);
      xs = linspace(xrange(1), xrange(2), npoints);
      p = exp(logprob(obj, xs(:)));
      h = plot(colvec(xs), colvec(p), plotArgs{:});
    end
    

  end
  

end