classdef TableJointDist < NonParamDist 
  % tabular (multi-dimensional array) distribution
  
  % No internal state

  
  %% main methods
  methods
    function m = TableJointDist()
    end
    %{
    function [X] = sample(obj, n)
       % X(i,:) = sample for i=1:n
       if nargin < 2, n = 1; end
       X = sampleDiscrete(obj.T, n, 1);
    end
    
    function x = mode(obj)
      x = argmax(obj.T);
    end
   %}
        
    
    function postQuery = marginal(eng, model, queryVars)
      % prob = sum_h p(Query,h)
      % Amortize cost of pre-processing across queries
      Tfac = dgmDiscreteToTfac(model);
      if ~iscell(queryVars)
        postQuery = marginalize(Tfac, queryVars);
      else
        for i=1:length(queryVars)
          postQuery{i} = marginalize(Tfac, queryVars{i});
        end
      end
    end
  
    
    %{
     function [obj, Z] = normalize(obj)
       % Ensure obj.T sums to 1
      [obj.T, Z] = normalize(obj.T);
     end
    %}
     
     function prob = predict(eng, model, visVars, visValues, queryVars)
      % computes p(Q|V=v)
      Tfac = dgmDiscreteToTfac(model);
      %domain = 1:ndimensions(model);
      prob = slice(Tfac, visVars, visValues); % p(H,v)
      prob = normalizeFactor(prob);
      if nargin >= 4
        prob = marginalize(prob, queryVars);
      end
     end

    %{
    function [ll, L] = logprob(obj, X)
      % ll(i) = log p(X(i,:) | params)
      % L = sum_i ll(i)
      sz = mysize(obj.T);
      ndx = subv2ind(sz, X); % convert data pattern into array index
      ll = log(obj.T(ndx));
      L = sum(ll);
    end
    %}
     
  end % methods

  methods(Static = true)
    function testSprinkler()
      [dgm] = DgmDist.mkSprinklerDgm;
      dgm.infEng = TableJointDist;
      Tfac = dgmDiscreteToTfac(dgm);
      J = Tfac.T; % CSRW
      C = 1; S = 2; R = 3; W = 4;
      pSgivenCW = predict(gdm, [C W], [1 1], [S]);
      pSgivenCW2 = sumv(J(1,:,:,1),3) ./ sumv(J(1,:,:,1),[2 3]);
      assert(approxeq(pSgivenCW.T(:), pSgivenCW2(:)))
    end
    
  end
  

end