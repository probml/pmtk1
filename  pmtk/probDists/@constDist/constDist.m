classdef constDist < probDist
  % vector of delta fns (constant values)
  properties
    x;
  end
 
  
  methods 
    function obj =  constDist(x)
      if nargin == 0
        x= [];
      end
      obj.x = x(:)';
      obj.names = {'x'};
    end
  
    function d = ndims(obj)
      d = length(obj.x);
    end
    
    function m = mean(obj)
      m = obj.x;
    end
    
    function m = mode(obj)
      m = obj.x;
    end  
    
    function m = var(obj)
      m = zeros(1, ndims(obj));
    end
    
     function X = sample(obj, n)
       % X(i,j) = sample from params(j)
       X = repmat(obj.x, n, 1);
     end
    
     function p = logProb(obj, X)
       % p(i,j) = log p(x(i) | params(j))
       d = ndims(obj);
       x = X(:);
       n = length(x);
       p = zeros(n,d);
       for j=1:d  
         p(:,j) = (x == obj.x(j));
       end
       p = log(p+eps);
     end
     
     function logZ = lognormconst(obj)
       logZ = 0;
     end
      
  end
    
end