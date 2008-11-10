classdef Dag < DirectedGraph 
  
  properties
    topoOrder;
  end


  methods
    function obj = Dag(varargin)
      % G = Dag(adjMat)
      % G = Dag('type', 'chain', 'nnodes', X)
      % G = Dag('type', 'rndDAGFanIn', 'nnodes', X, 'maxFanIn', X)
      % G = Dag('type', 'rndDAGEdgeProb', 'nnodes', X, 'edgeProb', X)
      if nargin == 0
        obj.adjMat = [];
        return;
      end
      if ~ischar(varargin{1})
        obj.adjMat = varargin{1};
      else
        [type, nnodes, maxFanIn, edgeProb] = process_options(...
          varargin, 'type', [], 'nnodes', [], 'maxFanIn', [], 'edgeProb', []);
        switch type
          case 'chain',
            % G(t,t+1) = 1 for all t<T
            obj.adjMat = diag(ones(1,nnodes-1),1);
          case 'rndDAGFanIn'
            obj.adjMat = mkRndDAGFanIn(nnodes, maxFanIn);
          case 'rndDAGEdgeProb'
            obj.adjMat = mkRndDAGEdgeProb(nnodes, edgeProb);
          otherwise
            error(['unrecognized arg ' varargin{1}])
        end
      end
      %obj.topoOrder = topological_sort(obj.adjMat);
      [d, pre, post, cycle, f, pred] = dfs(obj.adjMat, [], 1);
      if cycle
        warning('PMTK:Dag', 'not acyclic')
      end
      obj.topoOrder = post(end:-1:1);
    end

    function objs = mkAllDags(dummy, nnodes, order, loadFromFile)
      % Returns cell array of all DAGs on nnodes.
      % eg. dags = mkAllDags(dag(), 5);
      % If ~isempty(order), only generates DAGs in which node i
      % has parents from nodes in order(1:i-1).
      % Warning: the number of DAGs on d nodes is O(d! 2^(d choose 2))
      % See R. W. Robinson. Counting labeled acyclic digraphs, 1973.
      % Nnodes  2   3   4       5          6      7      8      9     10  
      % Ndags   3  25  543 29,281  3,781,503  1.1e9 7.8e11 1.2e15 4.2e18
      if nargin < 3, order = []; end
      if nargin < 4, loadFromFile = false; end
      if ~isempty(order), loadFromFile = false; end
      Gs = mk_all_dags(nnodes, order, loadFromFile);
      for i=1:length(Gs)
        objs{i} = Dag(Gs{i});
      end
    end
    
    function [M, moral_edges] = moralize(obj)
      % Ensure that for every child, all its parents are married (connected)
      % and then drop directionality of edges.
      M = obj.adjmat;
      n = length(M);
      for i=1:n
        fam = family(obj,i);
        M(fam,fam)=1;
      end
      M = setdiag(M,0);
      moral_edges = sparse(triu(max(0,M-obj.adjMat),1));
      M = UndirectedGraph(M);
    end
    
  end

end