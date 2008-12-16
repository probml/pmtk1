classdef BetaBinomDist < ParamDist
  
  properties
    a;
    b;
    N;
    support;
  end
 
  %% Main methods
  methods 
    function obj =  BetaBinomDist(N,a,b)
      % betabinomdist(N, a, b) 
      if nargin == 0
        N = []; a = []; b = [];
      end
      obj.a = a;
      obj.b = b;
      obj.N = N;
      obj.support = 0:N(1);
    end
 
    
    function d = ndistrib(obj)
      d = length(obj.a);
    end
    
    
    function m = mean(obj)
      m = obj.N * (obj.a ./(obj.a + obj.b));
    end
    
     function m = var(obj)
       a = obj.a; b = obj.b; N = obj.N;
       m = (N .* a .* b .* (a + b + N)) ./ ( (a+b).^2 .* (a+b+1) );
     end  
     
   
     function p = logprob(obj, X)
       % p(i,j) = log p(x(i) | params(j)), x(i) in 0:N
       d = ndistrib(obj);
       x = X(:);
       p = zeros(length(x),d);
       for j=1:d
         a = obj.a(j); b = obj.b(j); n = obj.N(j);
         p(:,j) = betaln(x+a, n-x+b) - betaln(a,b) + nchoosekln(n, x);
       end
     end
       
     function logZ = lognormconst(obj)
        d = ndistrib(obj);
        for j=1:d
          logZ(j) = betaln(obj.a, obj.b);
        end
     end
     
     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i,1)  = num of successes, data(i,2) = nu, failures
      % Uses Tom Minka's fixedpoint method
      [X] = process_options(...
        varargin, 'data', []);
      alphas = polya_fit_simple(X);
      obj.a = alphas(1); obj.b = alphas(2);
     end
    
     
     function h=plot(obj, varargin)
       ndistrib = length(obj.a);
       if ndistrib > 1, error('can only plot 1 distribution'); end
       [plotArgs] = process_options( varargin, 'plotArgs' ,{});
       if ~iscell(plotArgs), plotArgs = {plotArgs}; end
       h=bar(exp(logprob(obj,obj.support)), plotArgs{:});
       set(gca,'xticklabel',obj.support);
     end     
     
  end
    
end