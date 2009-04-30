classdef Binom_BetaDist < ParamDist % CompoundDist
 % p(X,theta|a,b,N) = Binom(X|N,theta) Beta(theta|a,b) 
  
 properties
   muDist;
   N;
   productDist;
 end
 
  %% Main methods
  methods 
    function obj =  Binom_BetaDist(varargin)
      % Binom_BetaDist(N, prior, productDist)
      % prior should be a BetaDist object
      [obj.N, obj.muDist, obj.productDist] = processArgs(varargin, ...
        '-N', [], '-prior', [], '-productDist', false);
    end

    function d = ndistrib(obj)
      d = ndistrib(obj.mu); % length(obj.N);
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
     
     function SS = mkSuffStat(obj,X) %#ok
       SS.nsucc = sum(X,1);
       ntrials = size(X,1);
       SS.nfail = ntrials*obj.N - SS.nsucc;
      end


     function obj = fit(obj, varargin)
       % m = fit(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       % data - X(i,1) is number of successes in N trials
       % suffStat - nsucc, nfail 
       [X, SS] = process_options(varargin,...
           'data'       , [],...
           'suffStat'   , []);
       if isempty(SS), SS = mkSuffStat(obj,X); end
       a = obj.muDist.a; b = obj.muDist.b;
       obj.muDist = BetaDist(a + SS.nsucc, b + SS.nfail);
       [n d] = size(X);
       if d>1 && isscalar(obj.N), obj.N = repmat(obj.N, 1, d); end
     end
           
  end 
 
end