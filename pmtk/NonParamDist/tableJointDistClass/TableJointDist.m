classdef TableJointDist < NonParamDist 
  % tabular (multi-dimensional array) distribution
  
  properties
    T;
    domain;
  end

  
  %% main methods
  methods
    function m = TableJointDist(T, domain)
      m.T  = T;
      if nargin < 2, domain = 1:ndims(T); end
      m.domain = domain;
    end

    function [X] = sample(obj, n)
       % X(i,:) = sample for i=1:n
       if nargin < 2, n = 1; end
       X = sampleDiscrete(obj.T, n, 1);
    end
    
    function d = ndimensions(obj)
     d = ndims(obj.T);
    end
  
    function x = mode(obj)
      x = argmax(obj.T);
    end
    
    function print(obj)
      disp('TableJointDist')
      dispjoint(obj.T);
    end
     
        
    function postQuery = marginal(obj, queryVars)
      R  = mysetdiff(obj.domain, queryVars); % variables to remove (marginalize over)
      Rndx = lookupIndices(R, obj.domain);
      smallT = obj.T;
      for i=1:length(Rndx)
        smallT = sum(smallT, Rndx(i));
      end
      smallT = squeeze(smallT);
      postQuery = TableJointDist(smallT, queryVars);
    end
    
     function Tsmall = slice(Tbig, visVars, visValues)
      % Return Tsmall(hnodes) = Tbig(visNodes=visValues, hnodes=:)
      if isempty(visVars), Tsmall = Tbig; return; end
      d = ndimensions(Tbig);
      Vndx = lookupIndices(visVars, Tbig.domain);
      ndx = mk_multi_index(d, Vndx, visValues);
      Tsmall = squeeze(Tbig.T(ndx{:}));
      H = mysetdiff(Tbig.domain, visVars);
      Tsmall = TableJointDist(Tsmall, H);
     end
    
     function [obj, Z] = normalize(obj)
       % Ensure obj.T sums to 1
      [obj.T, Z] = normalize(obj.T);
    end
    
     function prob = predict(obj, visVars, visValues, queryVars)
      % computes p(Q|V=v)
      prob = slice(obj, visVars, visValues); % p(H,v)
      prob = normalize(prob);
      if nargin >= 4
        prob = marginal(prob, queryVars);
      end
    end

    function [ll, L] = logprob(obj, X)
      % ll(i) = log p(X(i,:) | params)
      % L = sum_i ll(i)
      sz = mysize(obj.T);
      ndx = subv2ind(sz, X); % convert data pattern into array index
      ll = log(obj.T(ndx));
      L = sum(ll);
    end
    
  end % methods

  methods(Static = true)
    function testSprinkler()
      [dgm] = DgmDist.mkSprinklerDgm;
      T = dgmDiscreteToTable(dgm);
      J = T.T; % CSRW
      C = 1; S = 2; R = 3; W = 4;
      pSgivenCW = predict(T, [C W], [1 1], [S]);
      pSgivenCW2 = sumv(J(1,:,:,1),3) ./ sumv(J(1,:,:,1),[2 3]);
      assert(approxeq(pSgivenCW.T(:), pSgivenCW2(:)))
    end
    
  end
  

end