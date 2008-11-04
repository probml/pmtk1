classdef wishartDist  < matrixDist
  
  properties
  end
  
  %% main methods
  methods
    function m = wishartDist(dof, Sigma)
      % We require Sigma is posdef and dof > d-1
      if nargin == 0
        m.Sigma = [];
        return;
      end
      m.dof = dof;
      m.Sigma = Sigma; % scale maxtrix 
    end
    
    function objS = convertToScalarDist(obj)
      if ndims(obj) ~= 1, error('cannot convert to scalarDst'); end
      objS = gammaDist(obj.dof/2, obj.Sigma/2);
    end
     
    
    function d = ndims(obj)
      d = size(obj.Sigma,1);
    end
    
    function lnZ = lognormconst(obj)
      d = ndims(obj);
      v = obj.dof;
      S = obj.Sigma;
      lnZ = (v*d/2)*log(2) + mvtGammaln(d,v/2) +(v/2)*logdet(S);
    end
    
     function L = logprob(obj, X)
      % L(i) = log p(X(:,:,i) | theta)
      % If object is scalar, then L(i) = log p(X(i) | theta))
      v = obj.dof;
      d = ndims(obj);
      if d==1
        n = length(X);
        X(find(X==0)) = eps;
        X = reshape(X,[1 1 n]);
      else
        n = size(X,3);
      end
      logZ = lognormconst(obj);
      Sinv = inv(obj.Sigma);
      for i=1:n
        L(i) = (v-d-1)/2*logdet(X(:,:,i)) -0.5*trace(Sinv*X(:,:,i)) - logZ;
      end
      L = L(:);
     end
    
    function m = mean(obj)
      m = obj.dof * obj.Sigma;
    end
    
    function m = mode(obj)
      m = (obj.dof - ndims(obj) - 1) * obj.Sigma;
    end
    
    
    function X = sample(obj, n)
      % X(:,:,i) is a random matrix drawn from Wi() for i=1:n
      d = ndims(obj);
      if nargin < 2, n = 1; end
      X  = zeros(d,d,n);
      [X(:,:,1), D] = wishrnd(obj.Sigma, obj.dof);
      for i=2:n
        X(:,:,i) = wishrnd(obj.Sigma, obj.dof, D);
      end
    end
    
   function mm = marginal(obj, query)
      % If M ~ Wi(dof,S), then M(q,q) ~ Wi(dof, S(q,q))
      % Press (2005) p112
      q = length(query); d = ndims(obj); v = obj.dof;
      mm = wishartDist(v, obj.Sigma(query,query));
   end
    
   
    
    
  end
    
 
    
end