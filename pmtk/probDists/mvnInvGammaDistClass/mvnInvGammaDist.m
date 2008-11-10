classdef MvnInvGammaDist < VecDist
  % p(m,s2|params) = N(m|mu, s2 Sigma) IG(s2| a,b)
  properties
    mu;
    Sigma;
    a;
    b;
  end

  %% main methods
  methods
    function m = MvnInvGammaDist(varargin)
      if nargin == 0, varargin = {}; end
      [mu, Sigma, a, b] = process_options(...
        varargin, 'mu', [], 'Sigma', [], 'a', [], 'b', []);
      m.mu = mu; m.Sigma = Sigma; m.a = a; m.b = b;
    end
    
 
    function mm = marginal(obj, queryVar)
      % marginal(obj, 'sigma') or marginal(obj, 'mu')
      switch lower(queryVar)
        case 'sigma'
          mm = InvGammaDist(obj.a, obj.b);
        case 'mu'
          v = 2*obj.a;
          s2 = 2*obj.b/v;
          mm = MvtDist(v, obj.mu, s2*obj.Sigma);
        otherwise
          error(['unrecognized variable ' queryVar])
      end
    end
   

  end
    
  %% demos
  methods(Static = true)
  end
    
end