classdef LaplaceApproxDist  < MvnDist
   % Laplace approximation

   properties
     logZ;
   end
   
  methods
    function obj = LaplaceApproxDist(logdensityFn, initVal)
      if nargin == 0, obj; return;end
      [obj.mu, obj.Sigma, obj.logZ] = laplaceApprox(logdensityFn, initVal);
    end
    
    function logZ = lognormconst(obj)
      logZ = obj.logZ;
    end

  end
    
end