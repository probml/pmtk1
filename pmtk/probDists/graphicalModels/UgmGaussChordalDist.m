classdef UgmGaussChordalDist < UgmGaussDist
  % gaussian graphical model on decomposable graphs
  
  properties
      %mu;
      %Sigma;
      %G;
  end

  %%  Main methods
  methods
      function obj = UgmGaussChordalDist(G, mu, Sigma)
          % obj = UgmChordalGaussDist(G, HiwDist(...), []) uses a prior of the form
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
    
   
    
    
  end
  
  
  methods(Static = true)
    
    function objs = mkAllGgmDecomposable(delta, Phi)
      % objs{i} = ggmDecomp with HIW prior for i'th chordal graph
      % delta = dof, Phi = precmat of prior
      nnodes = size(Phi,1);
      Gs = ChordalGraph.mkAllChordal(nnodes, true);
      objs = cell(1, length(Gs));
      for i=1:length(Gs)
        objs{i} = UgmGaussChordalDist(Gs{i}, HiwDist(Gs{i}, delta, Phi), []);
      end
    end
    
  end
  
 
    

end