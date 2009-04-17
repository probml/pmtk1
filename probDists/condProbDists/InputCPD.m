classdef InputCPD < CondProbDist 
  % This represents a root node that is always observed
  
  properties 
  end
  
  methods
    
   function Tfac = convertToTabularFactor(CPD, child, ctsParents, dParents, visible, data, nstates,fullDomain) %#ok
     assert(isempty(ctsParents), 'no parents allowed for an inputCPD')
     assert(isempty(dParents), 'no parents allowed for an inputCPD')
     assert(visible(canonizeLabels(child,fullDomain)), 'node must be visible for an inputCPD')
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
