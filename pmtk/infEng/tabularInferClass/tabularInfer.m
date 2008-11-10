classdef TabularInfer < InfEngine
  % inference by exhaustive enumeration in multidimensional tables 
  
  properties
    Tfac;
    visVars = [];
    visValues = [];
    evidenceEntered = false;
  end
 
  methods
    function eng = TabularInfer(T)
      if nargin == 0, T = []; end
      sz = mysize(T);
      eng.Tfac = TabularFactor(T, 1:length(sz));
    end
    
    function eng = setParams(eng, params)
      T = params{1};
      eng.Tfac = TabularFactor(T, 1:ndims(T));
    end
    
     function [X, eng] = sample(eng, n)
       % X(i,:) = sample for i=1:n
       if ~eng.evidenceEntered
         eng = enterEvidence(eng, [], []); % compute joint
       end
       T = eng.Tfac.T(:);
       X = sampleDiscrete(T, n, 1);
     end
    
    function [postQuery, eng] = marginal(eng, queryVars)
      if ~eng.evidenceEntered
        eng = enterEvidence(eng, [], []);  % compute joint
      end
      smallpot = marginalize(eng.Tfac, queryVars, false);
      %postQuery = smallpot;
      postQuery = TabularDist(smallpot.T);
    end


    function eng = enterEvidence(eng, visVars, visValues)
      eng.visVars = visVars; eng.visValues = visValues;
      eng.evidenceEntered = true;
      % computes p(Xh|Xv=v) and stores internally
      Fnumer = slice(eng.Tfac, visVars, visValues); % p(H,v)
      Fnumer = normalizeFactor(Fnumer);
      eng.Tfac = Fnumer; % sets the domain to hidNodes
    end

    function [ll, L] = logprob(obj, X)
      % ll(i) = log p(X(i,:) | params)
      % L = sum_i ll(i)
      ndx = subv2ind(obj.Tfac.sizes, X);
      ll = log(obj.Tfac.T(ndx));
      L = sum(ll);
    end

  end % methods

end