classdef ChordalGraph < UndirectedGraph
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
      
      function obj = ChordalGraph(adjMat, action, varargin)
        % obj = ChordalGraph(adjMat, 'checkChordal')
        %     sets obj.ischordal = false if not chordal (in which case obj is
        %     invalid). This the default action.
        % obj = ChordalGraph(adjMat, 'makeChordal', 'elimOrder', X, 'nodeWeights', X)
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
          %warning('PMTK:ChordalGraph', 'graph is not chordal')
          return;
        end
        % number cliques so they satisfy running intersection property (RIP)
        [obj.cliques, cliquesNonRIP] = chordal_to_ripcliques_cell(obj.adjMat, obj.perfectElimOrder);
        [obj.seps, obj.resids, obj.hists] = seps_resids_hists_cell(obj.cliques);
        %obj.edgeStruct = makeEdgeStruct(double(obj.adjMat));
        obj = initEdgeStruct(obj);
      end
      
      
      
    end % methods
    
    methods(Static = true)
      function CGs = mkAllChordal(d, loadFromFile)
        % Returns cell array of all chordal graphs on nnodes.
        % Warning: the number of CGs on d nodes is exponential in d
        % See Helen Armstrong's PhD thesis, p149, U New South Wales 2005
        % Nnodes  2   3   4    5       6       7        8
        % Ncg     2   8   61   822  18,154  617,675  30,888,596
        if nargin < 2, loadFromFile = true; end

        fname = fullfile(PMTKroot(), 'data', sprintf('decompGraphs%d.mat', d));
        if loadFromFile && exist(fname, 'file')
          S = load(fname, '-mat');
          fprintf('loading %s\n', fname);
          CGs = S.CGs;
          return;
        end

        UGs = UndirectedGraph.mkAllUG(d, loadFromFile);
        CGs = {}; 
        for i=1:length(UGs)
          cg = ChordalGraph(UGs{i}.adjMat, 'checkChordal');
          if cg.ischordal
            CGs{end+1} = cg; %#ok
          end
        end
       
        if loadFromFile
          disp(['saving to ' fname]);
          save(fname, 'CGs');
        end
      end
      
    end
    
    
end