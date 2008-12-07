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
      % betabinomdist(N, a, b) where args are scalars
      if nargin == 0
        N = []; a = []; b = [];
      end
      obj.a = a;
      obj.b = b;
      obj.N = N;
      obj.support = 0:N;
    end
 
    function d = ndimensions(obj)
      d = length(obj.a);
    end
    
    function m = mean(obj)
      m = obj.N * (obj.a ./(obj.a + obj.b));
    end
    
     function m = var(obj)
       a = obj.a; b = obj.b; n = obj.N;
       m = (N .* a .* b .* (a + b + N)) ./ ( (a+b).^2 .* (a+b+1) );
     end  
     
   
     function p = logprob(obj, X, paramNdx)
       % p(i,j) = log p(x(i) | params(j))
       if nargin < 3, paramNdx = 1:ndimensions(obj); end
       x = X(:);
       p = zeros(length(x),length(paramNdx));
       for jj=1:length(paramNdx)
         j = paramNdx(jj);
         a = obj.a(j); b = obj.b(j); n = obj.N(j);
         p(:,jj) = betaln(x+a, n-x+b) - betaln(a,b) + nchoosekln(n, x);
       end
     end
       
     function logZ = lognormconst(obj)
       logZ = betaln(obj.a, obj.b);
     end
     
     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i) = case i
      % method - currently must be fixedpoint
      [X, suffStat, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'fixedpoint');
      alphas = polya_fit_simple(X);
      obj.a = alphas(1); obj.b = alphas(2);
     end
    
     
     function h=plot(obj, varargin)
         
         [plotArgs] = process_options( varargin, 'plotArgs' ,{});
         if ~iscell(plotArgs), plotArgs = {plotArgs}; end
         h=bar(exp(logprob(obj,obj.support)), plotArgs{:});
         set(gca,'xticklabel',obj.support);
         
     end
     
     
     
  end
    
end