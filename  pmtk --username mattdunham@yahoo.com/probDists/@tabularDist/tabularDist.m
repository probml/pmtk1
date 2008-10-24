classdef tabularDist < vecDist 
  % tabular (multi-dimensional array) distribution
  
  properties
    T;
  end
  
  %% main methods
  methods
    function m = tabularDist(T)
      if nargin == 0, T = []; end
      m.T = T;
      m.stateInfEng = tabularInfer;
    end

    function params = getModelParams(obj)
      params = {obj.T};
    end
    
    function obj = mkRndParams(obj)
      sz = mysizes(obj.T);
      obj.T = myreshape(rand(prod(sz),1), sz);
    end
    
    function d = ndims(m)
     d = ndims(m.T);
    end
  
    function [mu, m] = mode(m)
     [mu, m]= mode(m.stateInfEng);
    end
 
    function print(m)
      dispjoint(m.stateInfEng.Tfac.T); % assumes infEng is tabularInfer...
    end
    
  end % methods

  %% Demos
  methods(Static = true)

    function demoSprinkler()
      [G, CPD, joint, nodes] = makeSprinklerBN;
      % joint(c,s,r,w)
      C = nodes.C; R = nodes.R; S = nodes.S; W = nodes.W;
      T = tabularDist(joint);
      false = 1; true = 2;
      
      pSandW = sumv(joint(:,true,:,true), [C R]); % 0.2781
      pW = sumv(joint(:,:,:,true), [C S R]); % 0.6471
      pSgivenW = pSandW / pW;  % 0.4298
      
      mW = marginal(T, W);
      assert(approxeq(mW.T(true), 0.6471))
      mSW = marginal(T, [S W]);
      assert(approxeq(mSW.T(true,true), 0.2781))
    
      T1 = enterEvidence(T, W, true);
      T2 = marginal(T1, [S]);
      
      %{
      visVars = W; visValues = true; queryVars = S; obj = T;
        function [postQuery] = conditional(obj, visVars, visValues, queryVars)
      % p(Xh|Xvis=vis) doesn't change state of model
      obj = enterEvidence(obj, visVars, visValues);
      if nargin < 4, queryVars = mysetdiff(1:ndims(obj), visVars); end
      [postQuery] = marginal(obj, queryVars);
        end
    %}
      
      [pSgivenW2] = conditional(T, W, true, S)
      %assert(approxeq(pSgivenW, pSgivenW2(true)))
     %keyboard
     
      pRandW = sumv(joint(:,:,true,true), [C S]); % 0.4581
      pRgivenW = pRandW / pW; % 0.7079

      % P(R=t|W=t) > P(S=t|W=t), so
      % Rain more likely to cause the wet grass than the sprinkler

      pSandRandW = sumv(joint(:,true,true,true), [C]); % 0.0891
      pSgivenWR = pSandRandW / pRandW; % 0.1945

      % P(S=t|W=t,R=t) << P(S=t|W=t)
      % Sprinkler is less likely to be on if we know that
      % it is raining, since the rain can "explain away" the fact
      % that the grass is wet.
    end

  end

end