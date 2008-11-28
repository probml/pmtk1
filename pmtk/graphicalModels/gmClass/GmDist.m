classdef GmDist < ParamDist
  % graphical model
  
  properties
    G;
  end

  %%  Main methods
  methods
     
    function d = ndims(obj)
       d = nnodes(obj.G);
    end
    
    function logprob(obj,varargin)
        error('not yet implemented'); 
    end
   
  end
  
  %% Demos
  methods
   
  end

end