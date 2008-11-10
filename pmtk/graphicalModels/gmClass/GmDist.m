classdef GmDist < VecDist
  % graphical model
  
  properties
    G;
  end

  %%  Main methods
  methods
     
    function d = ndims(obj)
       d = nnodes(obj.G);
    end
    
  end
  
  %% Demos
  methods
   
  end

end