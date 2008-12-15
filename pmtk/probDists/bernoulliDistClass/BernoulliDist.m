classdef BernoulliDist < DiscreteDist
% A special case of the DiscreteDist for 2 states
  %% main methods
  methods
    function obj = BernoulliDist(mu)
      % mu(j) is success probablity for j'th distribution
      obj.support = [1,0];          % 1 for success, 0 for failure
      if nargin == 0;
          mu = []; 
      else
          mu = [mu(:)';1-mu(:)'];
      end
      obj.mu = mu;
    end
  end 
end

