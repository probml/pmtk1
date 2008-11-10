classdef Jtree < Tree
  % Junction (join) tree: nodes are the maxcliques in the corresponding chordal
  % graph
  
  properties
    cliques;
    seps;
    sepsize;
    fillInEdges;
  end

  methods
    function obj = Jtree(adjMat)
      % If adjmat is not chordal, it will be made so using the minweight
      % heuristic
      if nargin == 0
        obj.adjMat = [];
        return;
      end
      CG = ChordalGraph(adjMat, 'makeChordal');
      obj.cliques = CG.cliques;
      obj.fillInEdges = CG.fillInEdges;
      obj.adjMat = ripcliques_to_jtree_cell(obj.cliques);
      [obj.sepsize, obj.seps]=separators_cell(obj.cliques, obj.adjMat);
    end

  end

end