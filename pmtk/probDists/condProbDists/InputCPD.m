classdef InputCPD < CondProbDist 
  % This represents a root node that is always observed
  
  properties 
  end
  
  methods
    
   function Tfac = convertToTabularFactor(CPD, child, ctsParents, dParents, visible, data, nstates) %#ok
     assert(isempty(ctsParents))
     assert(isempty(dParents))
     assert(visible(child))
     Tfac = TabularFactor(1,child);  % return an empty TabularFactor
   end
   
   function p = isDiscrete(CPD) %#ok
     p = false;
   end

   function q = nstates(CPD)  %#ok
     q = 1;
   end
      
  end % methods
  
end
