classdef BernoulliDist < ParamDist
  
  % This differs from DiscreteDist on 2 states
  % since we only store the probabilities of heads,
  
  properties
    mu; % 1*d vector
  end
  
  %% main methods
  methods
    function obj = BernoulliDist(mu)
      obj.mu = mu;
    end
    
    
    function logp = logprob(obj, X)
       % p(i,j) = log p(X(i) | params(j))
       d = m.ndims;
       n = length(X);
       logp = zeros(n,d);
       X01 = canonizeLabels(X)-1; %convert to [0,1]
       for j=1:d
        logp(:,j) = X01*log(obj.mu(j)+eps) + (1-X01)*log(obj.mu(j)+eps);
       end
    end
     
  end
  
    %% Getters and Setters
  methods
      function obj = set.mu(obj, mu)
          obj.mu = mu;
          obj.ndims = length(mu);
      end 
  end
  
  methods(Static=true)
    function SS = mkSuffStat(obj,X)
      X01 = canonizeLabels(X)-1; %convert to [0,1]
      SS.counts = sum(X01); % sum over columns
    end
  end
         
  
end

