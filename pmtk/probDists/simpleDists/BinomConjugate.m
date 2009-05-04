classdef BinomConjugate < ProbDist
 % p(X,theta|a,b,N) = Binom(X|N,theta) Beta(theta|a,b) 
  
 properties
   muDist;
   N;
   productDist;
 end
 
  %% Main methods
  methods 
    function obj =  BinomConjugate(varargin)
      % BinomConjugate(N, prior, productDist)
      % prior should be a BetaDist object
      [obj.N, obj.muDist, obj.productDist] = processArgs(varargin, ...
        '-N', [], '-prior', [], '-productDist', false);
    end
    
    function m = marginalizeOutParams(obj)
      % p(X|a,b,N) = int_{theta} p(X,theta|a,b,N)
      a = obj.muDist.a; b = obj.muDist.b;
      m =  BetaBinomDist(obj.N, a, b, obj.productDist);
    end

     function p = logprob(obj, X)
       % p(i) = log p(X(i)) log marginal likelihood
       p = logprob(marginalizeOutParams(obj), X);
     end
     
     function SS = mkSuffStat(obj,X) 
       if isempty(X)
         SS.nsucc = 0; SS.nfail = 0;
       else
         SS.nsucc = sum(X,1);
         ntrials = size(X,1);
         SS.nfail = ntrials*obj.N - SS.nsucc;
       end
     end

     function obj = fit(obj, varargin)
       % m = fit(model, data, SS)
       % data(i,1) is number of successes in N trials
       % SS - sufficient statistics, struct with fields nsucc, nfail 
       % Examples
       % m = fit(m, X)
       % m = fit(m, '-SS', SS)
       [X, SS] = processArgs(varargin,...
           '-data'       , [],...
           '-SS'   , []);
       if isempty(SS), SS = mkSuffStat(obj,X); end
       a = obj.muDist.a; b = obj.muDist.b;
       obj.muDist = BetaDist(a + SS.nsucc, b + SS.nfail);
       [n d] = size(X);
       if d>1 && isscalar(obj.N), obj.N = repmat(obj.N, 1, d); end
     end
           
  end 
 
end