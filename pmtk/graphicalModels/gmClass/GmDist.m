classdef GmDist < ParamDist
  % graphical model
  
  properties
    G; %  a graph object
  end

  %%  Main methods
  methods
     
    function d = ndimensions(obj)
       d = nnodes(obj.G); % size(obj.G,1);
    end
   
  end
  
  %% Demos
  methods
   
  end

end