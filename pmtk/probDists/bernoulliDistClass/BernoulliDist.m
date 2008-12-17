classdef BernoulliDist < BinomDist

  methods
    function obj = BernoulliDist(mu)
      % mu(j) is success probablity for j'th distribution
      obj.support = [1,0];          % 1 for success, 0 for failure
      if nargin == 0, mu = []; end
      obj.mu = mu;
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

