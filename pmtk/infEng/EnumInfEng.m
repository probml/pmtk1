classdef EnumInfEng < InfEng 
  % tabular (multi-dimensional array) distribution
  % Model must support following method:
  % Tfac = convertToTabularFactor(model)
  
  properties
    Tfac;
    logZ;
  end

  
  %% main methods
  methods
    function eng = EnumInfEng()
      eng.Tfac = [];
    end

    function eng = condition(eng, model, visVars, visValues)
       Tfac = convertToTabularFactor(model);
       [Tfac, Z] = normalizeFactor(slice(Tfac, visVars, visValues)); % p(H,v)
       eng.Tfac = Tfac;
       eng.logZ = log(Z);
    end
    
    function [postQuery] = marginal(eng, queryVars)
      % postQuery = sum_h p(Query,h)
      if isempty(eng.Tfac), error('must first call condition'); end
      postQuery = marginalize(eng.Tfac, queryVars);
    end
     
     function [samples] = sample(eng,n)
       if nargin < 2, n = 1; end
       if isempty(eng.Tfac), error('must first call condition'); end
       [samples] = sample(eng.Tfac,  n);
     end
    
     function logZ = lognormconst(eng)
       logZ = eng.logZ;
      end
    
  end % methods

 
  

end