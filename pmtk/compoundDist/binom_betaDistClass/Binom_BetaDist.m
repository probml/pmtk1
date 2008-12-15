classdef Binom_BetaDist < ParamDist
 % p(X|a,b,N) = int Binom(X|N,theta) Beta(theta|a,b) dtheta
  
  %% Main methods
  methods 
      function obj =  BinomDist(N,mu)
      % binomdist(N,mu) binomial distribution
      % N is mandatory; mu can be omitted or set to [] if it will be
      % estimated (using fit).
      % mu can also be a random variable
      % eg pr = binomdist(10, betadist(1,1))    
          if nargin == 0;
              N = 0; mu = [];
          end
          if(numel(mu) == 1 && isnumeric(mu)),mu = [mu,1-mu];end
          obj.mu = mu;
          obj.N = N;
          obj.support = 0:N;
      end
    
    function d = ndimensions(obj)
      d = 1;
    end
    
    function h=plot(obj, varargin)
         
         [plotArgs] = process_options( varargin, 'plotArgs' ,{});
         if ~iscell(plotArgs), plotArgs = {plotArgs}; end
         h=bar(exp(logprob(obj,obj.support)), plotArgs{:});
         set(gca,'xticklabel',obj.support);
    end
    
    function X = sample(obj, n)
        X = sum( rand(n,obj.N) < repmat(obj.mu(1), n, obj.N), 2);
     end
    
     function p = logprob(obj, X)
     % p(i) = log(p(X(i)|model))
     % eg., logprob(binomdist(10,[0.5 0.1]), 1:10)
         X = X(:);
         if useStatsToolbox
             p = log(binopdf(X, obj.N, obj.mu(1)));
         else
             % LOG1P  Compute log(1+z) accurately.
             p = nchoosekln(obj.N, X) + X.*log(obj.mu(1)) + (obj.N - X).*log1p(-obj.mu(1));
         end
         
     end
     
     function pr = predict(obj)
       % p(X|D) 
       switch class(obj.params)
         case 'BetaDist' % integrrate out mu
           pr = BetaBinomDist(obj.N, obj.params.a, obj.params.b);
         otherwise
           pr = obj;
       end
     end

  end 
 
end