classdef GmDist < ParamJointDist
  % graphical model
  
  properties
    G; %  a graph object
    %infEng;
  end

  %%  Main methods
  methods
     
    function d = ndimensions(obj)
       d = nnodes(obj.G); % size(obj.G,1);
    end
    
    function d = nnodes(obj)
       d = nnodes(obj.G); % size(obj.G,1);
    end
    
    function h = drawGraph(obj)
        h = draw(obj.G);
    end

  end
  

end