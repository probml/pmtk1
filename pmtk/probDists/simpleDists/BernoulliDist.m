classdef BernoulliDist < BinomDist

  methods
    function obj = BernoulliDist(varargin)
      % mu(j) is success probablity for j'th distribution
      [obj.mu, obj.prior, obj.support, obj.productDist] = processArgs(varargin, ...
        '-mu', [], '-prior', NoPrior, '-support',[1,0], '-productDist', false);
      % 1 for success, 0 for failure
      obj.N = 1;
    end
  end
  
end

