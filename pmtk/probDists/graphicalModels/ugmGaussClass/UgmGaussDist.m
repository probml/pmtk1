classdef UgmGaussDist < UgmDist
    % undirected gaussian graphical model
    
    properties
        mu;
        Sigma;
        precMat;
    end
    
    %%  Main methods
    methods
      function obj = UgmGaussDist(G, mu, Sigma)
        % GgmDist(G, mu, Sigma) where G is of type graph
        % mu and Sigma can be [] and set later.
        if nargin < 1, G = []; end
        if nargin < 2, mu = []; end
        if nargin < 3, Sigma = []; end
        if isa(G, 'double'), G = UndirectedGraph(G); end
        obj.G = G;
        obj.mu = mu; obj.Sigma = Sigma;
        obj.domain = 1:length(mu);
        obj.infMethod = GaussInfEng();
      end
      
      
      function [mu,Sigma,domain] = convertToMvn(m)
        mu = m.mu; Sigma = m.Sigma;
        domain = 1:length(m.mu);
      end

      
      %{
      function postQuery = marginal(model, queryVars, visVars, visValues)
        % we currently ignore the graph strucutre
        %[mu,Sigma,domain] = convertToMvnDist(model);
        mvn = MvnDist(model.mu, model.Sigma, '-domain', model.domain);
        postQuery = marginal(mvn, queryVars, visVars, visValues);
      end
      %}
      
      function obj = mkRndParams(obj)
        % Set Sigma to a random pd matrix such that SigmaInv is consistent
        % with G
        d = ndimensions(obj);
        obj.mu = randn(d,1);
        A = obj.G.adjMat;
        prec = randpd(d) .* A;
        obj.precMat = mkDiagDominant(prec);
        % density = 0.8; reciprocal_condition = 0.9;
        %Prc_true = full(sprandsym(D,density,reciprocal_condition,2));
        obj.Sigma = inv(obj.precMat);
      end

      function L = logprob(obj, X)
        % L(i) = log p(X(i,:) | params)
        L = logprob(MvnDist(obj.mu, obj.Sigma), X);
      end

      
      function np = dof(model)
        d = ndimensions(model);
        np = nedges(model.G) + d; % num elts in precmat, plus mu
      end

      function obj = fit(obj, varargin)
        % Point estimate of parameters given graph
        % m = fit(m, X, SS)
        % 
        % X(i,:) = case i
        % SS - sufficient statistics, SS.S, SS.mu, SS.N
        [X, SS, shrink] = processArgs(varargin, '-X', [], '-SS', [], '-shrink', false);
        if isa(X, 'DataTable'), X = X.X; end
        if isempty(SS)
          SS.mu = mean(X);
          SS.N = size(X,1);
          if shrink
            SS.S = covshrink(X);
          else
            SS.S = cov(X);
          end
        end
        if nnodes(obj.G)==0, obj = fitStructure(obj, '-SS', SS); end
        obj.mu = SS.mu;
        [obj.precMat, iter] = ggmFitHtf(SS.S, obj.G.adjMat); %#ok
        obj.Sigma = inv(obj.precMat);
      end

      function obj = fitStructure(obj, varargin)
        % Find best graph structure
        % m = fitStructure(m, X, SS, method, lambda, warmstartCov)
        %
        % Method can be one 'glasso', 'glassoR' [glasso]
        % X(i,:) is data
        % SS - sufficient statistics, SS.S, SS.mu, SS.N
        [X, SS, method, lambda, W] = processArgs(...
          varargin, '-X', [], '-SS', [], '-method', 'glasso', '-lambda', 1e-3,  ...
          '-warmstartCov', []);
        if isa(X, 'DataTable'), X = X.X; end
        if isempty(SS)
          SS.mu = mean(X);
          SS.N = size(X,1);
          SS.S = cov(X);
        end
        obj.mu = SS.mu;
        switch method
          case 'glasso',
            if isempty(W)
              [obj.precMat, obj.Sigma] = ggmLassoHtf(SS.S, lambda);
            else
               [obj.precMat, obj.Sigma] = ggmLassoHtf(SS.S, lambda, 'W', W);
            end
          case 'glassoR',
            [obj.precMat, obj.Sigma] = ggmLassoR(SS.S, lambda);
          otherwise
            error(['unknown method ' method])
        end
        obj.G = UndirectedGraph(precmatToAdjmat(obj.precMat));
      end

      function X = sample(obj, n)
        % X(i,:) = i'th case
        % CUrrently we ignore the graph structure
        X = sample(MvnDist(obj.mu, obj.Sigma), n);
      end


      function d = nnodes(obj)
        % num dimensions (variables)
        d = nnodes(obj.G);
      end

    end

end