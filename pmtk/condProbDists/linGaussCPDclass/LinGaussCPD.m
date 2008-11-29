classdef LinGaussCPD < CondProbDist 
  % p(y|x) = N(y | Wx + mu, Sigma) where x is all the parents
  
  properties 
    W;
    mu;
    Sigma;
    domain;
  end
  
  methods
    function obj = LinGaussCPD(domain, W, mu, Sigma)
      % domain = indices of each parent, followed by index of child
      obj.domain = domain;
      obj.W = W; obj.mu = mu; obj.Sigma = Sigma;
    end
    
  end
  
end
