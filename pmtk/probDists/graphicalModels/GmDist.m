classdef GmDist 
  % graphical model
  
  properties
    G; %  a graph object
    domain;
    infMethod;
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
         if(~iscell(queryVars))
             model = removeBarrenNodes(model,queryVars,visVars);
         end
        [eng, logZ, other] = condition(model.infMethod, model, visVars, visVals);
        if ~iscell(queryVars)
            [postQuery] = marginal(eng, queryVars);
        else
            for q=1:length(queryVars)
                postQuery{q} = marginal(eng, queryVars{q}); %#ok
            end
        end
    end
       
    
   
  end
 
  methods(Access = 'protected')
      
      function model = removeBarrenNodes(model,queryVars,visVars)     
      % We remove barren nodes from directed models, that is, unobserved 
      % nodes with exactly one parent and no children. If this results in a
      % new barren node by this definition, this too is removed, 
      % recursively. Note, this just gets the low lying fruit as it were;
      % there are many more nodes we could potentially prune by further
      % conditional independance tests. 
      
          if ~isdirected(model) || isempty(queryVars) || numel(queryVars)==numel(model.domain), return; end  
          map    = @(x)canonizeLabels(x,model.domain);  % maps from domain to 1:d
          invmap = @(x)model.domain(x);                 % maps from 1:d to domain
          
          hidden = setdiff(model.domain,union(queryVars,visVars));
          remove = false(1,numel(model.domain));
          vstructs = invmap(sum(model.G.adjMat,1) ~= 1);       % don't remove roots or nodes with multiple parents
          remaining = setdiff(hidden,vstructs);
          while not(isempty(remaining))
              consider = map(remaining(1));
              d = descendants(model.G.adjMat,consider);
              if all(ismember(invmap(d),hidden)) && ~any(ismember(invmap(d),vstructs)) % works in case d=[]
                remove(consider) = true; 
                remove(d) = true;
                remaining = setdiff(remaining(2:end),invmap(d));
              else
                  remaining(1) = [];
              end
          end
          model = removeNodes(model,invmap(remove));
      end
      
      
      
  end

end