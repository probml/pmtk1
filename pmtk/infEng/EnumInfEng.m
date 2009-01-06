classdef EnumInfEng < InfEng 
  % tabular (multi-dimensional array) distribution
  % Model must support following method:
  % Tfac = convertToTabularFactor(model)
  
  properties
    Tfac;
  end

  
  %% main methods
  methods
    function eng = EnumInfEng()
      eng.Tfac = [];
    end

    function eng = condition(eng, model, visVars, visValues)
       Tfac = convertToTabularFactor(model);
       Tfac = normalizeFactor(slice(Tfac, visVars, visValues)); % p(H,v)
       eng.Tfac = Tfac;
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
    
  end % methods

  methods(Static = true)
    function testClass()
      [dgm] = mkSprinklerDgm;
      dgm.infEng = EnumInfEng;
      Tfac = convertToTabularFactor(dgm);
      J = Tfac.T; % CSRW
      C = 1; S = 2; R = 3; W = 4;
      dgm = condition(dgm, [C W], [1 1]);
      pSgivenCW = marginal(dgm, S);
      pSgivenCW2 = sumv(J(1,:,:,1),3) ./ sumv(J(1,:,:,1),[2 3]);
      assert(approxeq(pSgivenCW.T(:), pSgivenCW2(:)))
      X = sample(dgm, 100);
    end
    
  end
  

end