classdef hiwDist < matrixDist
  
  properties
    G;
    delta;
    Phi;
  end
  
  %% main methods
  methods
    function m = hiwDist(G, delta, Phi)
      % hiwDist(G, delta, Phi) G is a decomposableGraph, delta a scalar,
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
      % Based on Roverato 2000, Prop 2
      % Written by Helen Armstrong (see her thesis p63)
      % Modified by Kevin Murphy
      ln_prod_top_terms=0;
      cliques = obj.G.cliques; seps = obj.G.seps;
      delta = obj.delta; Phi = obj.Phi;
      for i=1:length(cliques)
        C_i=cliques{i};
        Phi_C_i=Phi(C_i, C_i);
        numC_i=length(C_i);
        ln_top_term_i= ( (delta+ numC_i -1) /2 ) * log(det(Phi_C_i/2) )...
          - mvt_gamma_ln( numC_i, (delta+ numC_i -1) /2);
        ln_prod_top_terms=ln_prod_top_terms+ln_top_term_i;
      end
      ln_prod_bottom_terms=0;
      for i=1:length(seps)
        S_i=seps{i};
        numS_i=length(S_i);
        Phi_S_i=Phi(S_i, S_i);
        ln_bottom_term_i=( (delta+ numS_i -1) /2 )* log(det(Phi_S_i/2))...
          - mvt_gamma_ln( numS_i, (delta+ numS_i -1) /2);
        ln_prod_bottom_terms=ln_prod_bottom_terms+ln_bottom_term_i;
      end
      ln_h=ln_prod_top_terms-ln_prod_bottom_terms;
      %assert(isequal(ln_h, lognormconst2(obj)))
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
    
  end
  
 


end