classdef UnifDist < ScalarDist 
  %  continuous uniform distribution
  
  properties
    lo;
    hi;
  end
  
  %% Main methods
  methods
    function m = UnifDist(lo, hi)
      if nargin == 0
        lo = []; hi =[];
      end
     m.lo = lo; m.hi = hi;
    end

    function d = ndims(m)
      d = length(m.lo);
    end
    
    
    function L = logprob(obj, X)
      % L(i,j) = log p(X(i) | params(j))
      [N d] = size(X);
      d = ndims(obj);
      if d==1, X = X(:); end
      L = zeros(N,d);
      for j=1:d
        valid = find( (X(:,j) >= obj.lo(j)) & (X(:,j) <= obj.hi(j)) );
        %prob(valid,j) = 1/(obj.hi(j)-obj.lo(j));
        L(valid,j) = -log(obj.hi(j)-obj.lo(j));
        L(~valid,j) = NaN;
      end
    end
    
   
   
     function X = sample(obj, n)
      % X(i,j) = sample ffrom params(j) i=1:n
      d = ndims(obj);
      assert(statsToolboxInstalled);
      for j=1:d
        X(:,j) = unifrnd(obj.lo(j), obj.hi(j), n, 1);
      end
    end

    

  end % methods

  %% Demos
  methods
    
  end
  
  
end