classdef InvGammaDist < ProbDist
    

  properties
    a;
    b;
  end

  %% Main methods
  methods
    function obj =  InvGammaDist(varargin)
      % a = shape, b = scale
      [obj.a, obj.b] = processArgs(varargin, ...
        '-a', [], '-b', []);
    end
      
    

    function d = ndimensions(obj)
      d = length(obj.a);
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
      d = ndimensions(obj);
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
      d = ndimensions(obj);
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
  
  
  methods
      
      function xrange = plotRange(obj) % over-ride default
          xrange = [0 3*mean(obj)];
      end
      
  end

end