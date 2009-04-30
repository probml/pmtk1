classdef BetaBinomDist < ParamDist
  
  properties
    a;
    b;
    N;
    support;
    productDist;
  end
 
  %% Main methods
  methods 
    function obj =  BetaBinomDist(varargin)
      % betaBinomdist(N, a, b, productDist)
      [obj.N, obj.a, obj.b, obj.productDist] = processArgs(varargin, ...
        '-N', [], '-a', [], '-b', [], '-productDist', false);
      if ~isempty(obj.N), obj.support = 0:obj.N(1); end
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
     
   
     function L = logprob(obj, X)
       % Return column vector of log probabilities for each row of X
      % L(i) = log p(X(i) | params)
      % L(i) = log p(X(i) | params(i)) (set distrib)
      % L(i) = sum_j log p(X(i,j) | params(j)) (prod distrib)
      % where X(i,j) in 0:N(j)
       n = size(X,1);
       a = obj.a; b = obj.b; N = obj.N;
       if ~obj.productDist
         X = X(:);
         if isscalar(a)
           a = repmat(a, n, 1); b = repmat(b, n, 1); N = repmat(N, n, 1);
         end
        L = betaln(X+a, N-X+b) - betaln(a,b) + nchoosekln(N, X);
       else
         d = length(a);
         Lij = zeros(n,d);
         for j=1:d
           Lij(:,j) = betaln(X(:,j)+a(j), N(j)-X(:,j)+b(j)) - ...
             betaln(a(j),b(j)) + nchoosekln(N(j), X(:,j));
         end
         L = sum(Lij,2);
       end
     end
       
     function logZ = lognormconst(obj)
       a = obj.a; b = obj.b;
       d = length(a);
       logZ = zeros(1,d);
       for j=1:d
         logZ(j) = betaln(a(j), b(j));
       end
     end
     
     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i,1)  = num of successes, data(i,2) = num failures
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
       h=bar(exp(logprob(obj,obj.support')), plotArgs{:});
       set(gca,'xticklabel',obj.support);
     end     
     
  end
    
end