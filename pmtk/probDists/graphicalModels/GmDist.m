classdef GmDist 
  % graphical model
  
  properties
    G; %  a graph object
    domain;
    infEng;
    discreteNodes;
    ctsNodes;
    nstates; % nstates(i) is number of discrete values, or size of vector valued node
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
      
  
    function [postQuery, logZ, other] = marginal(model, queryVars, visVars, visVals)
      if nargin < 3, visVars = []; visVals = []; end
      [eng, logZ, other] = condition(model.infEng, model, visVars, visVals);
      if ~iscell(queryVars)
        [postQuery] = marginal(eng, queryVars);
      else
        for q=1:length(queryVars)
          postQuery{q} = marginal(eng, queryVars{q}); %#ok
        end
      end
    end
       
   
  end
 

end