classdef UgmChordalGaussDist < UgmDist
  % gaussian graphical model on decomposable graphs
  
  properties
  end

  %%  Main methods
  methods
    function obj = GgmDecomposableDist(G, mu, Sigma)
      % obj = GgmDecomposableDist(G, HiwDist(...), []) uses a prior of the form
      % p(mu) propto 1, p(Sigma) = hiw(G)
       if nargin == 0
         G = []; mu = []; Sigma = [];
       end
       obj.G = G; obj.mu = mu; obj.Sigma = Sigma;
    end
    
    function d = nnodes(obj)
      if ~isempty(obj.G), d = nnodes(obj.G); return; end
      if isa(obj.mu, 'HiwDist')
        d = size(obj.mu.Phi,1); 
      end
    end
    
    function L = logmarglik(obj, varargin)
      % L = logmarglik(obj, 'param1', name1, ...)
      % Arguments:
      % 'data' - data(i,:) is value for i'th case
      [Y] = process_options(varargin, 'data', []);
      assert(isa(obj.mu, 'HiwDist'))
      % p77 of Helen Armstrong's PhD thesis eqn 4.16
      n = size(Y,1);
      Sy = n*cov(Y,1);
      G = obj.G; delta = obj.mu.delta; Phi = obj.mu.Phi;
      nstar = n-1; d = ndimensions(obj);
      L = lognormconst(HiwDist(G, delta, Phi)) ...
        - lognormconst(HiwDist(G, delta+n, Phi+Sy)) ...
        -  (nstar*d/2) * log(2*pi);
    end
    
    function objs = mkAllGgmDecomposable(obj)
      % objs{i} = ggmDecomp with HIW prior for i'th chordal graph
      assert(isa(obj.mu, 'HiwDist'))
      delta = obj.mu.delta; Phi = obj.mu.Phi;
      nnodes = size(Phi,1);
      Gs = mkAllChordal(ChordalGraph, nnodes, true);
      for i=1:length(Gs)
        objs{i} = GgmDecomposableDist(Gs{i}, HiwDist(Gs{i}, delta, Phi), []);
      end
    end
    
    function [logpostG, GGMs, mapG, mapPrec, postG, postMeanPrec, postMeanG] = ...
        computePostAllModelsExhaustive(obj, Y)
      GGMs = mkAllGgmDecomposable(obj);
      N = length(GGMs);
      prior = normalize(ones(1,N));
      logpostG = zeros(1,N);
      for i=1:N
        logpostG(i) = log(prior(i)) + logmarglik(GGMs{i}, 'data', Y);
      end
      bestNdx = argmax(logpostG);
      mapG = GGMs{bestNdx}.G;
      n = size(Y,1);
      nstar = n-1; % since mu is unknown
      Sy = n*cov(Y,1);
      delta = obj.mu.delta; Phi = obj.mu.Phi;
      deltaStar = delta + nstar; PhiStar = Phi + Sy;
      mapPrec = meanInverse(HiwDist(mapG, deltaStar, PhiStar));
      if nargout >= 3
        logZ = logsumexp(logpostG(:));
        postG = exp(logpostG - logZ);
      end
      if nargout >= 4
        d = nnodes(obj);
        postMeanPrec = zeros(d,d);
        postMeanG = zeros(d,d);
        % Armstrong thesis p80
        for i=1:N
          postMeanPrec = postMeanPrec + postG(i) * meanInverse(HiwDist(GGMs{i}.G, deltaStar, PhiStar));
          postMeanG = postMeanG + postG(i) * GGMs{i}.G.adjMat;
        end
      end
    end
    
    
  end
  
 
    

end