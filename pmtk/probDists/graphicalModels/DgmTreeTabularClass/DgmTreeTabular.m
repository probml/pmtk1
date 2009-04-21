classdef DgmTreeTabular < DgmDist 
    % directed tree with tabular potentials
    
    properties
        %G; infMethod;
        nodeStateSpace; % we assume each node has the same state space
    end
    
    %%  Main methods
    methods
        function obj = DgmTreeTabular(varargin)
            % UgmTreeTabular(...)
            % 'G' - graph structure (adjacency matrix or RootedTree object)
            % 'CPDs' - cell array of tabularCPD
            % 'infMethod' - [JtreeInfEng]
            % nstates(j) - number of states for node j
            % nodeStateSpace - vector of integers eg [0,1]
            if(nargin == 0);return;end
            [G, obj.CPDs, obj.infMethod, obj.nstates, obj.nodeStateSpace] = process_options(varargin, ...
                'G', [], 'CPDs', [], 'infEng', JtreeInfEng(), 'nstates', [], 'nodeStateSpace', []);
            if isa(G, 'double'), G = RootedTree(G, 1); end
            obj.G = G;
            obj.domain = 1:nnodes(G);
            obj.discreteNodes = 1:nnodes(G);
        end
       
        
        function model = fitStructure(model, varargin)
          % Find MLE tree using Chow-Liu algorithm
          % 'data' - X(i,j) is value of node j in case i, i=1:n, j=1:d
          [X, model.nodeStateSpace] = process_options(varargin, ...
            'data', [], 'nodeStateSpace', model.nodeStateSpace);
          if isempty(model.nodeStateSpace)
            model.nodeStateSpace = unique(unique(full(X)));
          end
          %[n d] = size(X);
          [treeAdjMat] = chowLiuTree(X, model.nodeStateSpace);
          root = 1; % arbitrary
          model.G = RootedTree(triu(treeAdjMat), root);
        end
        
        function model = fit(model, varargin)
          % Find the MAP estimate of the parameters of the CPTs.
          % If the structure is unknown, find the MLE structure first.
           % 'data' - X(i,j) is value of node j in case i, i=1:n, j=1:d
          [X, dirichlet] = process_options(varargin, ...
            'data', [], 'dirichlet', 0);
          if nnodes(model.G)==0, model = fitStructure(model, 'data', X); end
          [N d] = size(X);
          sz = length(model.nodeStateSpace)*ones(1,d); 
          X = canonizeLabels(X); % 1...K requried by compute_counts
          for i=1:d
            pa = parents(model.G, i);
            if isempty(pa) % no parent
              cnt = compute_counts(X(:,i)', sz(i));
              model.CPDs{i} = normalize(cnt+dirichlet);
            else
              j = pa;
              cnt = compute_counts(X(:,[j i])', sz([j i])); % parent then child
              model.CPDs{i} = mkStochastic(cnt+dirichlet);
            end
          end
        end
        
        function ll = logprob(model, X)
          % LL(n) = log p(X(n,:) | model.params)
          [N d] = size(X);
          X = canonizeLabels(X); % 1...K, used to index into CPTs
          ll = zeros(N,1);
          for i=1:d
            j = parents(model.G, i);
            CPT = model.CPDs{i};
            if isempty(j)
              ll = ll + log(CPT(X(:,i))+eps);
            else
              ndx = sub2ind(size(CPT), X(:,j), X(:,i)); % parent then child
              ll = ll + log(CPT(ndx)+eps);
            end
          end
        end
        
        function plotGraph(model, varargin)
          [nodeLabels] = process_options(varargin, 'nodeLabels', 1:nnodes(model.G));
          Graphlayout('adjMatrix', full(model.G.adjMat), ...
            'nodeLabels', nodeLabels, 'currentLayout', Treelayout());
        end

        
    end % methods 
end