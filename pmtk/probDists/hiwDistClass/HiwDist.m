classdef HiwDist < ProbDist
  
  properties
    G;
    delta;
    Phi;
  end
  
  %% main methods
  methods
    function m = HiwDist(G, delta, Phi)
      % HiwDist(G, delta, Phi) G is a decomposableGraph, delta a scalar,
      % Phi is a positive definite matrix
      if nargin == 0
        m.G = [];
        return;
      end
      m.G = G;
      m.delta = delta;
      m.Phi = Phi;
    end
    
    function d = ndims(obj)
      d = size(obj.Phi,1); 
    end
    
    
    function ln_h = lognormconst(obj)
      ln_h = HIWlognormconst(obj);
    end
    
    
    
   
    function m = meanInverse(obj)
      % E[inv(Sigma)]
      % See Armstrong thesis p80
      d = ndims(obj);
      m = zeros(d,d);
      cliques = obj.G.cliques; seps = obj.G.seps;
      delta = obj.delta; Phi = obj.Phi;
      for i=1:length(cliques)
        C_i=cliques{i};
        Phi_C_i=Phi(C_i, C_i);
        numC_i=length(C_i);
        m(C_i,C_i) = m(C_i,C_i) + (delta + numC_i - 1)*inv(Phi_C_i);
      end
      for i=1:length(seps)
        C_i=seps{i};
        Phi_C_i=Phi(C_i, C_i);
        numC_i=length(C_i);
        m(C_i,C_i) = m(C_i,C_i) - (delta + numC_i - 1)*inv(Phi_C_i);
      end
    end
      
    function X = sample(obj, n)
      % X(:,:,i) is a random matrix drawn from HIW() for i=1:n
      d = ndims(obj);
      X  = zeros(d,d,n);
      for i=1:n
        Sigma_id = sampleFromIdentity(obj.G, obj.delta);
        [Sigma_D, K_D] = transformSample(obj.G, Sigma_id, obj.delta, eye(d), obj.Phi);
        X(:,:,i) = Sigma_D;
      end
    end
    
 
  function logprob(obj,varargin)
      error('not yet implemented');
  end
  
  
  end
end