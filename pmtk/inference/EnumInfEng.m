classdef EnumInfEng < InfEng 
  % tabular (multi-dimensional array) distribution
  % Model must support following method:
  % Tfac = convertToJointTabularFactor(model)
  
  properties
    Tfac;
  end

  
  %% main methods
  methods
    function eng = EnumInfEng()
      eng.Tfac = [];
    end

    function [eng, logZ, other] = condition(eng, model, visVars, visValues)
      if ~isempty(model.ctsNodes)
        error('EnumInfEng requires all nodes to be discrete')
      end
       Tfac = convertToJointTabularFactor(model);
       [Tfac, Z] = normalizeFactor(slice(Tfac, visVars, visValues)); % p(H,v)
       eng.Tfac = Tfac;
       logZ = log(Z);
       other = [];
    end
    
    function [postQuery] = marginal(eng, queryVars)
      % postQuery = sum_h p(Query,h)
      if isempty(eng.Tfac), error('must first call condition'); end
      postQuery = marginalize(eng.Tfac, queryVars);
    end
    
    function X = sample(eng,n)
        error('Sampling is not implemented for EnumInfEng, use JtreeInfEng instead');
    end
         
  end % methods

end