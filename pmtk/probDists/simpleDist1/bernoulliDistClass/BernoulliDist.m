classdef BernoulliDist < BinomDist

  methods
    function obj = BernoulliDist(varargin)
      % mu(j) is success probablity for j'th distribution
      [obj.mu, obj.prior,obj.support] = process_options(varargin, ...
        'mu', [], 'prior', 'none','support',[1,0]);
      % 1 for success, 0 for failure
      obj.N = 1;
    end
  end
  
  methods(Static = true)
    function testClass()
      m = BernoulliDist;
      X = rand(10,2)>0.5;
      m = fit(m, 'data', X);
    end
  end
end

