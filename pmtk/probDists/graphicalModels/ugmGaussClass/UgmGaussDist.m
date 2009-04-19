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
        mvn = MvnDist(model.mu, model.Sigma, 'domain', model.domain);
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
        obj.Sigma = inv(obj.precMat);
      end

      function L = logprob(obj, X)
        % L(i) = log p(X(i,:) | params)
        L = logprob(MvnDist(obj.mu, obj.Sigma), X);
      end

      function B = bicScore(obj, X)
        % B = log p(X|model) - (dof/2)*log(N);
        N = size(X,1);
        dof = nedges(obj.G);
        L = logprob(MvnDist(obj.mu, obj.Sigma), X);
        B = sum(L) - (dof/2)*log(N);
      end

      function obj = fit(obj, varargin)
        % Point estimate of parameters given graph
        % m = fit(model, 'name1', val1, 'name2', val2, ...)
        % Arguments are
        % data - data(i,:) = case i
        % 'SS' - sufficient statistics, SS.C, SS.mu
        [X, SS] = process_options(...
          varargin, 'data', [], 'suffStat', []);
        if nnodes(obj.G)==0, obj = fitStructure(obj, 'data', X); end
        if isempty(SS)
          SS.mu = mean(X);
          SS.N = size(X,1);
          SS.S = cov(X);
        end
        obj.mu = SS.mu;
        [obj.precMat, iter] = ggmFitHtf(SS.S, obj.G.adjMat); %#ok
        obj.Sigma = inv(obj.precMat);
      end

      function obj = fitStructure(obj, varargin)
        [method, lambda, X, W] = process_options(...
          varargin, 'method', 'glasso', 'lambda', 1e-3, 'data', [], ...
          'warmstartCov', []);
        S = cov(X);
        obj.mu = mean(X);
        switch method
          case 'glasso',
            if isempty(W)
              [obj.precMat, obj.Sigma] = ggmLassoHtf(S, lambda);
            else
               [obj.precMat, obj.Sigma] = ggmLassoHtf(S, lambda, 'W', W);
            end
          case 'glassoR',
            [obj.precMat, obj.Sigma] = ggmLassoR(S, lambda);
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