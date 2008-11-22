classdef MvnInvWishartDist < VecDist
  % p(m,S|params) = N(m|mu, 1/k * S) IW(S| dof, Sigma)
  properties
    mu;
    Sigma;
    dof;
    k;
  end

  %% main methods
  methods
    function m = MvnInvWishartDist(varargin)
      if nargin == 0, varargin = {}; end
      [mu, Sigma, dof, k] = process_options(...
        varargin, 'mu', 0, 'Sigma', 1, 'dof', 1, 'k', []);
      d = length(mu);
      if isempty(k), k = d; end
      m.mu = mu; m.Sigma = Sigma; m.dof = dof; m.k = k;
    end
    
    function d = ndims(obj)
      % for the purposes of plotting etc, the dimensionality is |m|+|S|
      d = length(obj.mu) + length(obj.Sigma(:));
    end
    
    function lnZ = lognormconst(obj)
      d = length(obj.mu); 
      v = obj.dof;
      S = obj.Sigma;
      k = obj.k;
      lnZ = (v*d/2)*log(2) + mvtGammaln(d,v/2) -(v/2)*logdet(S) + (d/2)*log(2*pi/k);
    end
     
    function L = logprob(obj, X)
      % For 1d, L(i) = log p(X(i,:) | theta), where X(i,:) = [mu sigma]
      d = length(obj.mu);
      if d > 1, error('not supported'); end
      pgauss = MvnDist(obj.mu, obj.Sigma/obj.k); % obj.Sigma is the argument!!
      warning('wrong formula')
      piw = InvWishartDist(obj.dof, obj.Sigma);
      n = size(X,1);
      assert(size(X,2)==2);
      L = logprob(pgauss, X(:,1)) + logprob(piw, X(:,2));
    end
      
    
    function mm = marginal(obj, queryVar)
      % marginal(obj, 'Sigma') or marginal(obj, 'mu')
      switch lower(queryVar)
        case 'sigma'
          d = size(obj.Sigma,1);
          if d==1 
            mm = InvGammaDist(obj.dof/2, obj.Sigma/2);
          else
            mm = InvWishartDist(obj.dof, obj.Sigma);
          end
        case 'mu'
          d = length(obj.mu);
          v = obj.dof;
          if d==1
            mm = StudentDist(v, obj.mu, obj.Sigma/(obj.k*v));
          else
            mm = MvtDist(v-d+1, obj.mu, obj.Sigma/(obj.k*(v-d+1)));
          end
        otherwise
          error(['unrecognized variable ' queryVar])
      end
    end
    
   
    

  end
  
  methods
      function xrange = plotRange(obj)
          d = length(obj.mu);
          if d > 1, error('not supported'); end
          sf = 2;
          S = obj.Sigma/obj.k;
          xrange = [obj.mu-sf*S, obj.mu+sf*S, 0.01, sf*S];
      end
  end
    
    
end