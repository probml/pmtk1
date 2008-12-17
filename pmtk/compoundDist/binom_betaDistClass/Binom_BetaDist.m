classdef Binom_BetaDist < CompoundDist
 % p(X,theta|a,b,N) = Binom(X|N,theta) Beta(theta|a,b) 
  
 properties
   muDist;
   N;
 end
 
  %% Main methods
  methods 
    function obj =  Binom_BetaDist(N,muDist)
      % Binom_Betadist(N,muDist) where muDist is of type BetaDist
      obj.N = N;
      obj.muDist = muDist;
    end

    function d = ndistrib(obj)
      d = length(obj.N);
    end
    
    function m = marginal(obj)
      a = obj.muDist.a; b = obj.muDist.b;
      m =  BetaBinomDist(obj.N, a, b);
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
     end
           
  end 
 
end