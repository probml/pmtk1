classdef UgmGaussDist < GmDist
  % undirected gaussian graphical model 
  
  properties
    mu;
    Sigma;
  end

  %%  Main methods
  methods
    function obj = UgmGaussDist(G, mu, Sigma)
      % GgmDist(G, mu, Sigma) where G is of type graph
      % mu and Sigma can be [] and set later.
      if nargin < 1, G = []; end
      if nargin < 2, mu = []; end
      if nargin < 3, Sigma = []; end
      obj.G = G; obj.mu = mu; obj.Sigma = Sigma;
      obj.domain = 1:length(mu);
      obj.infEng = GaussInfEng; % ignores graph structure
    end

    function [mu,Sigma,domain] = convertToMvnDist(m)
      mu = m.mu; Sigma = m.Sigma;
      domain = 1:length(m.mu); 
    end
    
     
    %{
    function params = getModelParams(obj)
      params = {obj.mu, obj.Sigma, obj.G};
    end
%}
    
    function obj = mkRndParams(obj)
      % Set Sigma to a random pd matrix such that SigmaInv is consistent
      % with G
      d = ndimensions(obj);
      obj.mu = randn(d,1);
      A = obj.G.adjMat;
      prec = randpd(d) .* A;
      prec = mkDiagDominant(prec);
      obj.Sigma = inv(prec);
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
      [X, SS] = process_options(...
        varargin, 'data', [], 'suffStat', []);
      obj.mu = mean(X);
      [precMat, covMat] = gaussIPF(cov(X), obj.G.adjMat);
      obj.Sigma = covMat;
    end

    function obj = fitStructure(obj, varargin)
      [method, lambda, X] = process_options(...
        varargin, 'method', 'L1BCD', 'lambda', 1e-3, 'data', []);
      switch method
        case 'L1BCD', [precMat, covMat] = L1precisionBCD(X, 'regularizer', lambda);
          obj.mu = mean(X);
          obj.Sigma = covMat;
          obj.G = UndirectedGraph(precmatToAdjmat(precMat));
        otherwise
          error(['unknown method ' method])
      end
    end

    function X = sample(obj, n)
      % X(i,:) = i'th case
      X = sample(MvnDist(obj.mu, obj.Sigma), n);
    end


    function d = nnodes(obj)
      % num dimensions (variables)
      d = nnodes(obj.G);
    end

  end

end