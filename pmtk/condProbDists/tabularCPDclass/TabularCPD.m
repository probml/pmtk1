classdef TabularCPD < CondProbDist 

  properties 
    T;
    domain;
    sizes;
  end
  
  methods
    function obj = TabularCPD(T, domain)
      % domain = indices of each parent, followed by index of child
      obj.T = T;
      obj.domain = domain;
      obj.sizes = mysize(T);
    end
    
    function Tfac = convertToTabularFactor(obj)
      Tfac = TabularFactor(obj.T, obj.domain);
    end
  end
  
end
