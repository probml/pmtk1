classdef BernoulliDist < DiscreteDist
% A special case of the DiscreteDist
  %% main methods
  methods
    function obj = BernoulliDist(mu)
      
      obj.support = [1,0];          % 1 for success, 0 for failure
      if nargin == 0;
          mu = []; 
      elseif(numel(mu) == 1 && isnumeric(mu))
          mu = [mu,1-mu];
      elseif(numel(mu) > 2)
         error('Use a DiscreteDist or BernoulliProductDist instead - mu must be of size 2 to use a BernoulliDist'); 
      end
      obj.mu = mu;
    end
  end
  
end

