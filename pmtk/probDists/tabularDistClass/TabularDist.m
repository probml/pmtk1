classdef TabularDist < VecDist 
  % tabular (multi-dimensional array) distribution
  
  properties
    T;
  end
  
  %% main methods
  methods
    function m = TabularDist(T)
      if nargin == 0, T = []; end
      m.T = T;
      m.stateInfEng = TabularInfer;
    end

    function params = getModelParams(obj)
      params = {obj.T};
    end
    
    function obj = mkRndParams(obj)
      sz = mysizes(obj.T);
      obj.T = myreshape(rand(prod(sz),1), sz);
    end
    
    function d = ndims(m)
     d = ndims(m.T);
    end
  
    function [mu, m] = mode(m)
     [mu, m]= mode(m.stateInfEng);
    end
 
    function print(m)
      dispjoint(m.stateInfEng.Tfac.T); % assumes infEng is tabularInfer...
    end
    
  end % methods

  

end