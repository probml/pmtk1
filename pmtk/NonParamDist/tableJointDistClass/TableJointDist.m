classdef TableJointDist < NonParamDist 
  % tabular (multi-dimensional array) distribution
  
  properties
    T;
  end

  
  %% main methods
  methods
    function m = TableJointDist(T)
      m.T  = T;
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
      dom = 1:ndimensions(obj);
      H  = mysetdiff(dom, queryVars);
      smallT = obj.T;
      for i=1:length(H)
        smallT = sum(smallT, H(i));
      end
      smallT = squeeze(smallT);
      postQuery = TableJointDist(smallT);
    end
    
     function Tsmall = slice(Tbig, vNodes, visValues)
      % Return Tsmall(hnodes) = Tbig(visNodes=visValues, hnodes=:)
      if isempty(vNodes), Tsmall = Tbig; return; end
      d = ndimensions(Tbig);
      ndx = mk_multi_index(d, vNodes, visValues);
      Tsmall = squeeze(Tbig.T(ndx{:}));
      Tsmall = TableJointDist(Tsmall);
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
      ndx = subv2ind(sz, X);
      ll = log(obj.T(ndx));
      L = sum(ll);
    end
    
  end % methods

  

end