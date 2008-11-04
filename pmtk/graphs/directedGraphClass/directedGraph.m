classdef directedGraph < graph 
  
  properties
    %adjMat;
    %biog;
    %topoOrder;
  end

  methods
    function obj = directedGraph(varargin)
      % G = directedGraph(adjMat)
      if nargin == 0
        obj.adjMat = [];
        return;
      end
      G = varargin{1};
      obj.adjMat = G;
      %  we allow self loops so checkAcyclic can use reachabilityGraph
      %obj.adjMat = setdiag(G,0);
    end


    function e = nedges(obj)
      e =  sum(obj.adjMat(:));
    end

    function obj2 = reachabilityGraph(obj)
      % obj2(i,j) = 1 iff there is a path from i to j in G
      % Transitive closure

      % expm(G) = I + G + G^2 / 2! + G^3 / 3! + ...
      M = expm(double(full(obj.adjMat))) - eye(length(obj.adjMat));
      obj2 = directedGraph(M>0);
    end

    function ps = parents(obj, i)
      ps = find(obj.adjMat(:,i))';
    end

    function cs = children(obj, i)
      cs = find(obj.adjMat(i,:));
    end

    function f = family(obj, i)
      % FAMILY Return the indices of parents and self in sorted order
      f = [parents(obj,i) i];
    end

    function b = checkAcyclic(obj)
      R = reachabilityGraph(obj);
      b = ~any(diag(R.adjMat)==1); % can't get back to yourself
    end

  end

end