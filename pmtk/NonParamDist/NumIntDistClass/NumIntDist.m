classdef NumIntDist  < NonParamDist
   % Compute posterior quantities using numerical integration (up to 3d)
   
  properties
    densityFn;
    logDensityFn;
    range;
    normconst;
    tol;
  end

  methods
    function obj = NumDist(logDensityFn, range, tol)
      % densityFn(X(i,:)) is the unnormalized density at specifed vector
      % range = [x1min, x1max, x2min, x2max, x3min, x3max]
      if nargin < 3, tol = 1e-5; end
      obj.densityFn = @(X) exp(logDensityFn(X));
      obj.logDensityFn = logDensityFn;
      obj.range = range;
      obj.tol = tol;
      obj.normconst = numericalIntegral(obj.densityFn, range, tol);
    end
    
    function logZ = lognormconst(obj)
      logZ = log(obj.normconst);
    end
    
    function d = ndimensions(obj)
      d = length(obj.range)/2;
    end
    
  
    
    function m = moment(obj, pow)
      for dim=1:ndimensions(obj)
        Z = obj.normconst;
        fn = @(X) X(:,dim).^pow;
        expectedfn = @(X) (obj.densityFn(X)/Z) .* fn(X);
        m(dim) =  numericalIntegral(expectedfn, obj.range, obj.tol);
      end
    end
    
     function m = mean(obj)
       m = moment(obj, 1);
     end
    
     function v = var(obj)
       % For 2d, returns marginal variances v = [var(X1); var(X2)]
       m1 = moment(obj, 1);
       m2 = moment(obj, 2);
       v = m2 - m1.^2;
     end
    
    function m = mode(obj)
      d = ndimensions(obj);
      range = reshape(obj.range, 2, d);
      start = mean(range);
      m =  maxFuncNumerical(obj.logDensityFn, start); 
    end

    
    function logp = logprob(obj, X)
      logp = log(obj.densityFn(X) - log(obj.normconst));
    end
    
    
  end

 
  
  methods(Static = true)
    function testClass()
      disp('foo')
    end
  end
  
end