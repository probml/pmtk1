classdef InputCPD < CondProbDist 
  % This represents a root node that is always observed
  
  properties 
  end
  
  methods
    
   function Tfac = convertToTabularFactor(model, domain,visVars,visVals) %#ok
     Tfac = TabularFactor(1,domain); return; % return an empty TabularFactor
   end
          
  end % methods
  
end
