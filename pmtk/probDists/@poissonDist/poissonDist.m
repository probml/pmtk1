classdef poissonDist < discreteDist
  
  properties
    lambda;
  end
 
  
  methods 
    function obj =  poissonDist(lambda)
      if nargin == 0;
        lambda = [];
      end
      obj.lambda = lambda;
      obj.support = 0:25; % truncate support for plotting purposes
    end

    function d = ndims(obj)
      d = length(obj.lambda);
    end
    
    function m = mean(obj)
     checkParamsAreConst(obj)
      m = obj.lambda;
    end
    
    function m = mode(obj)
      checkParamsAreConst(obj)
      m = floor( obj.lambda );
    end  
    
    function m = var(obj)
      checkParamsAreConst(obj)
       m = obj.lambda;
    end
     
   
    function X = sample(obj, n)
       % X(i,j) = sample from params(j) for i=1:n
       checkParamsAreConst(obj)
       d = ndims(obj);
       X = zeros(n, d);
       if ~statsToolboxInstalled, error('need stats toolbox'); end
       for j=1:d
         X(:,j) = poissrnd(obj.lambda(j), n, 1);
       end
     end
    
     function p = logprob(obj, X, paramNdx)
       % p(i,j) = log p(x(i) | params(j))
       checkParamsAreConst(obj)
       d = ndims(obj);
       if nargin < 3, paramNdx = 1:d; end
       x = X(:);
       n = length(x);
       p = zeros(n,length(paramNdx));
       for jj=1:length(paramNdx)
         j = paramNdx(jj);
         p(:,jj) = x .* log(obj.lambda(j)) - factorialln(x) - obj.lambda(j);
       end
     end


     function logZ = lognormconst(obj)
       logZ = -obj.lambda;
     end
        
     

  end
  %% Private methods
  methods(Access = 'protected')

    function checkParamsAreConst(obj)
      p = isa(obj.lambda, 'double');
      if ~p
        error('parameters must be constants')
      end
    end

  end
  
end