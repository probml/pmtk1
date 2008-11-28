classdef BernoulliDist < BinomDist
% A special case of the binomial distribution.   
  %% main methods
  methods
    function obj = BernoulliDist(mu)
      
      obj.support = [0,1];
      if nargin == 0;
        mu = [];
      end
      
     obj = setup(obj, 1, mu, true);
    end
  end
  
end

