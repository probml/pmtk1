classdef UndirectedGraph < Graph 
  
  properties
    edgeStruct;
  end

%% main methods
  methods
    function obj = UndirectedGraph(varargin)
      % G = UndirectedGraph(adjMat)
      % G = UndirectedGraph('type', 'rnd', 'nnodes', X, 'edgeProb', X)
      % G = UndirectedGraph('type', 'chain', 'nnodes', X)  Markov chain
      % G = UndirectedGraph('type', loop', 'nnodes', X)
      % G = UndirectedGraph('type', 'lattice2D', 'nrows', X, 'ncols', X,
      %            'wrapAround', X, 'connectivity', X)
      %    where 'connectivity' is one of(4,8,12 or 24, def 4)
      obj.directed = false;
      if nargin == 0
        obj.adjMat = [];
        return;
      end
      if ~isstr(varargin{1})
        G = varargin{1};
      else
        [type, edgeProb, nnodes, nrows, ncols, connectivity, ...
          wrapAround, maxFanIn] = process_options(...
          varargin, 'type', [], 'edgeProb', 0.1, 'nnodes', [], ...
          'nrows', [], 'ncols', [], 'connectivity', 4, 'wrapAround', false, 'maxFanIn', []);
        if isempty(maxFanIn), maxFanIn = nnodes; end
        switch lower(type)
          case 'rnd',
            G = rand(nnodes, nnodes) < edgeProb;
            G = mkGraphSymmetric(G);
          case 'chain',
            G = diag(ones(1,nnodes-1),1) + diag(ones(1,nnodes-1),-1);
          case 'loop'
            G = tridiag(ones(nnodes, nnodes));
            G(1,nnodes) = 1; G(nnodes, 1) = 1;
          case 'lattice2d',
            if wrapAround
              assert(connectivity==4);
              if nnodes > 100
                fprintf('warning: slow!\n');
              end
              G = mk2DLatticeWrap(nrows, ncols, wrapAround);
            else
              G = mk2DLatticeNoWrap(nrows, ncols, connectivity);
            end
          case 'aline4'
            % 1-2
            %  \/
            %  3 - 4
            G = [0     1     1     0;
              1     0     1     0;
              1     1     0     1;
              0     0     1     0];
          otherwise
            error(['unknown type ' type])
        end % switch type
      end
      obj.adjMat = mkGraphSymmetric(G);
      obj.adjMat = setdiag(obj.adjMat,0);
      %obj.edgeStruct = makeEdgeStruct(double(obj.adjMat));
      obj = initEdgeStruct(obj);
    end

     function objs = mkAllUG(dummy, nnodes, loadFromFile)
      % Returns cell array of all UGs on nnodes.
      % eg. UGs = mkAllUG(UndirectedGraph(), 5);
      % Warning: the number of UGs on d nodes is O(2^(d choose 2))
      % which is just the number of ways to label every possible edge.
      % Nnodes  2   3   4  5       6       7        8      9     10  
      % Nug     2   8  64  1024  32,678 2.1e6   2.7e8 6.9e10 3.5e13
      if nargin < 3, loadFromFile = true; end
      Gs = mk_all_ugs(nnodes, loadFromFile);
      for i=1:length(Gs)
        objs{i} = UndirectedGraph(Gs{i});
      end
    end

    function e = nedges(obj)
      e = obj.edgeStruct.nEdges;
    end

    function e = edgeBetween(obj, n1, n2)
      E = obj.edgeStruct.edgeEnds;
      ndx1 = find(E(:,1)==n1);
      ndx2 = find(E(:,2)==n2);
      e = intersect(ndx1, ndx2);
    end

    function ns = edgeEnds(obj, e)
      % ns = [n1 n2]
      ns = obj.edgeStruct.edgeEnds(e,:);
    end
    
    function edges = edgesFromNode(obj, n)
      E = obj.edgeStruct.E;
      V = obj.edgeStruct.V;
      edges = E(V(n):V(n+1)-1);
    end

    function obj2 = reachabilityGraph(obj)
      % obj2(i,j) = 1 iff there is a path from i to j in G
      % Transitive closure

      % expm(G) = I + G + G^2 / 2! + G^3 / 3! + ...
      M = expm(double(full(obj.adjMat))) - eye(length(obj.adjMat));
      obj2 = UndirectedGraph(M>0);
    end

    function clqs = maximalCliques(obj)
      % clqs{i} are the node's in the i'th clique
      % This algorithm can take exponential time in the worst case
     clqs = maximalCliques(obj.adjMat);
    end
   
    function b = checkAcyclic(obj)
      % e.g., G =
      % 1 -> 3
      %      |
      %      v
      % 2 <- 4
      % In this case, 1->2 in the transitive closure, but 1 cannot get to itself.
      % If G was undirected, 1 could get to itself, but this graph is not cyclic.
      % So we cannot use the closure test in the undirected case.
      [d, pre, post, cycle] = dfs(obj);
      b = ~cycle;
    end
    
     function [obj2, cost] = minSpanTree(obj)
     % minimum weight spanning tree, where obj.adjMat(i,j) is the weight from i->j
     % Set obj.adjMat = -1*obj.adjMat first to find max spanning tree.
     % Uses Prim's algorithm, which is O(d^2)
     [A, cost] = minimum_spanning_tree(obj.adjMat);
     obj2 = Tree(A);
     end

  end
  
  %% Operator overload
  methods
    % we overload the syntax so obj(i,j)= obj(j,i) = obj.adjMat(i,j) 
    function obj2 = subsasgn(obj, S, value)
      if (numel(S) > 1) % eg. obj.adjMat(1:3,:) = value
        obj2 = builtin('subsasgn', obj, S, value);
      else
        switch S.type    %eg obj(1:3,:)
          case {'()'}
            obj2 = obj;
            obj2.adjMat(S.subs{1}, S.subs{2}) = value;
            obj2.adjMat(S.subs{2}, S.subs{1}) = value; % new (enforce symmetry)
          case '.' % eg. obj.adjMat = value
            obj2 = builtin('subsasgn', obj, S, value);
        end
      end
    end
    
    function val = subsref(obj, S)
      if (numel(S) > 1) % eg. obj.adjMat(1:3,:)
        val = builtin('subsref', obj, S);
      else
        switch S.type    %eg obj(1:3,:)
          case {'()'}
            val = obj.adjMat(S.subs{1}, S.subs{2});
          case '.' % eg. obj.adjMat = value
            val = builtin('subsref', obj, S);
        end
      end
    end
   
  end
  
  
  %% Private methods
  methods(Access = 'protected')
    % we want to make private functions available to child classes
    % so we make them methods  
    function obj = initEdgeStruct(obj)
      obj.edgeStruct = makeEdgeStruct(double(obj.adjMat));
    end
  end
  
end