classdef chordalGraph < undirectedGraph  
  % chordal (decomposable/ triangulated) graphs
  
  properties(GetAccess = 'public', SetAccess = 'protected')
    perfectElimOrder;
    cliques;
    seps;
    resids;
    hists;
    fillInEdges;
    ischordal;
  end

  %% main methods
  methods
    
    function obj = chordalGraph(adjMat, action, varargin)
      % obj = chordalGraph(adjMat, 'checkChordal') 
      %     sets obj.ischordal = false if not chordal (in which case obj is
      %     invalid). This the default action.
      % obj = chordalGraph(adjMat, 'makeChordal', 'elimOrder', X, 'nodeWeights', X)
      %    makes adjMat chordal by adding extra edges if necessary.
      %    elimOrder is one of {'minFill', 'minWeight'}
      %     where mcs = max cardinaltiy search. (Default is minWeight.)
      %    If elimOrder = minWeight, you can specify nodeWeights
      %    which, for discrete rv's, is log the number of states.
      %    If weights = ones(1,d) (Default), we minimize the induced clique size.
      if nargin == 0
        obj.adjMat = [];
        return;
      end;
      obj.adjMat = mkGraphSymmetric(adjMat);
      if nargin < 2, action = 'checkChordal'; end
      if strcmp(action, 'makeChordal')
        d = length(adjMat);
        [elimOrder, nodeWeights] = process_options(...
          varargin, 'elimOrder', 'minFill', 'nodeWeights', ones(1,d));
        if strcmp(elimOrder, 'minFill')
          nodeWeights = zeros(1,d);
        end
        order = minweightElimOrder(adjMat, nodeWeights);
        [obj.adjMat, obj.fillInEdges] = mk_chordal(obj.adjMat, order);
      end
      [obj.ischordal, obj.perfectElimOrder] = check_chordal(obj.adjMat);
      if ~obj.ischordal
        %warning('BLT:chordalGraph', 'graph is not chordal')
        return;
      end
      % number cliques so they satisfy running intersection property (RIP)
      [obj.cliques, cliquesNonRIP] = chordal_to_ripcliques_cell(obj.adjMat, obj.perfectElimOrder);
      [obj.seps, obj.resids, obj.hists] = seps_resids_hists_cell(obj.cliques);      
      %obj.edgeStruct = makeEdgeStruct(double(obj.adjMat));
      obj = initEdgeStruct(obj);
    end

  function objs = mkAllChordal(dummy, nnodes, loadFromFile)
      % Returns cell array of all chordal graphs on nnodes.
      % eg. CGs = mkAllUG(chordalGraph(), 5);
      % Warning: the number of CGs on d nodes is exponential in d
      % See Helen Armstrong's PhD thesis, p149, U New South Wales 2005
      % Nnodes  2   3   4    5       6       7        8     
      % Ncg     2   8   61   822  18,154  617,675  30,888,596
      if nargin < 3, loadFromFile = true; end
      Gs = mkAllUG(undirectedGraph(), nnodes, loadFromFile);
      objs = {};
      for i=1:length(Gs)
        cg = chordalGraph(Gs{i}.adjMat, 'checkChordal');
        if cg.ischordal
          objs{end+1} = cg;
        end
      end
  end

   
 
  end % methods
  
  %% demos
  methods(Static = true)
    
    function demo()
      chordalGraphDemo; % stored in private directory for brevity
    end

  end


end