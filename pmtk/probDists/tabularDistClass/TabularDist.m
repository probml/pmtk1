classdef TabularDist < VecDist 
  % tabular (multi-dimensional array) distribution
  
  properties
    Tfac;
  end
  
  %% main methods
  methods
    function m = TabularDist(T)
      if nargin == 0, T = []; end
      sz = mysize(T);
      m.Tfac = TabularFactor(T, 1:length(sz));
    end

    function [X] = sample(obj, n)
       % X(i,:) = sample for i=1:n
       T = obj.Tfac.T(:);
       X = sampleDiscrete(T, n, 1);
    end
    
    function d = ndims(obj)
     d = ndims(obj.Tfac);
    end
  
    function m = mode(obj)
      m = mode(obj.Tfac);
    end
 
    function print(obj)
      dispjoint(obj.Tfac.T);
    end
     
    function postQuery = marginal(obj, queryVars)
      smallpot = marginalize(obj.Tfac, queryVars, false);
      %postQuery = smallpot;
      postQuery = TabularDist(smallpot.T);
    end
    
     function postQuery = predict(obj, visVars, visValues, queryVars)
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