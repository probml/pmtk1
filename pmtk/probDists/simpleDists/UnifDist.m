classdef UnifDist < ParamDist 
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

    function d = ndimensions(m)
      d = length(m.lo);
    end
    
    
    function L = logprob(obj, X)
      % L(i,j) = log p(X(i) | params(j))
      [N d] = size(X);
      d = ndimensions(obj);
      if d==1, X = X(:); end
      L = zeros(N,d);
      for j=1:d
        valid = find( (X(:,j) >= obj.lo(j)) & (X(:,j) <= obj.hi(j)) );
        %prob(valid,j) = 1/(obj.hi(j)-obj.lo(j));
        L(valid,j) = -log(obj.hi(j)-obj.lo(j));
        L(~valid,j) = NaN;
      end
    end
    
    function m = mean(obj)
        m = (obj.hi + obj.lo)/2;
    end
    
    function m = mode(obj)
        m = obj.lo;
    end
    
    function v = var(obj)
       v = ((obj.hi - obj.lo).^2)/12;
    end
   
   
     function X = sample(obj, n)
      % X(i,j) = sample ffrom params(j) i=1:n
      d = ndimensions(obj);
      assert(statsToolboxInstalled);
      for j=1:d
        X(:,j) = unifrnd(obj.lo(j), obj.hi(j), n, 1);
      end
     end
    
     function e = entropy(obj)
        e = log(obj.hi - obj.lo);
     end
        
     function m = momentGeneratingFunction(obj)
        m = @(t)(exp(t*obj.hi)-exp(t*obj.lo))/(t*(obj.hi-obj.lo)); 
     end
    

  end % methods

  %% Demos
  methods
    
  end
  
  
end