classdef betaBinomDist < discreteDist
  
  properties
    a;
    b;
    N;
  end
 
  %% Main methods
  methods 
    function obj =  betaBinomDist(N,a,b)
      % betabinomdist(N, a, b) where args are scalars
      if nargin == 0
        N = []; a = []; b = [];
      end
      obj.a = a;
      obj.b = b;
      obj.N = N;
      obj.support = 0:N;
    end
 
    function d = nfeatures(obj)
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
       if nargin < 3, paramNdx = 1:nfeatures(obj); end
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
     
     function obj = inferParams(obj, varargin)
      % m = inferParams(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i) = case i
      % method - currently must be fixedpoint
      [X, suffStat, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'fixedpoint');
      alphas = polya_fit_simple(X);
      obj.a = alphas(1); obj.b = alphas(2);
    end
  end
    
  %% Demos
  methods(Static = true)
    function demoEbCancer()
      % Johnson and Albert  p67, p24
      data.y = [0 0 2 0 1 1 0 2 1 3 0 1 1 1 54 0 0 1 3 0];
      data.n = [1083 855 3461 657 1208 1025 527 1668 583 582 917 857 ...
        680 917 53637 874 395 581 588 383];

      % EB matrix of counts
      X = [data.y(:) data.n(:)-data.y(:)];
      dist = betaBinomDist;
      dist = inferParams(dist, 'data', X, 'method', 'fixedpoint');
      a = dist.a; b = dist.b;
      [a b]
      
      d = length(data.n); % ncities;
      for i=1:d
        aPost(i) = a + data.y(i);
        bPost(i) = b + data.n(i) - data.y(i);
        thetaPostMean(i) = aPost(i)/(aPost(i) + bPost(i));
        thetaMLE(i) = data.y(i)/data.n(i);
      end
      thetaPooledMLE = sum(data.y)/sum(data.n);

      figure;
      subplot(4,1,1); bar(data.y); title('number of people with cancer (truncated at 5)')
      set(gca,'ylim',[0 5])
      subplot(4,1,2); bar(data.n); title('pop of city (truncated at 2000)');
      set(gca,'ylim',[0 2000])
      subplot(4,1,3); bar(thetaMLE);title('MLE');
      subplot(4,1,4); bar(thetaPostMean);title('posterior mean (red line=pooled MLE)')
      hold on;h=line([0 20], [thetaPooledMLE thetaPooledMLE]);
      set(h,'color','r','linewidth',2)

    end
    
  end

end