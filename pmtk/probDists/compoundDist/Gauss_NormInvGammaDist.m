classdef Gauss_NormInvGammaDist < CompoundDist
  % p(X,mu,sigma2|m,k,a,b) = N(X|mu, sigma2) NIG(mu,sigma2| m,k,a,b)
  
  properties
    muSigmaDist;
  end
  
  %% Main methods
  methods 
     function m = Gauss_NormInvGammaDist(prior)
      m.muSigmaDist = prior;
     end
     
     function obj = fit(obj, varargin)
       [X, SS] = process_options(...
         varargin, 'data', [], 'suffStat', []);
       xbar = mean(X);
       [n d] = size(X);
       m0 = obj.muSigmaDist.mu;
       k0 = obj.muSigmaDist.k;
       a0 = obj.muSigmaDist.a;
       b0 = obj.muSigmaDist.b;
       kn = k0 + n; 
       mn = (k0*m0 + n*xbar)/kn;
       an = a0 + n/2;
       XC = X - repmat(xbar,n,1);
       bn = b0 + 0.5*sum(XC.^2) + 0.5*n*k0*(m0-xbar).^2./(k0+n);
       if d>1 && isscalar(an)
         an = repmat(an, 1, d);
         kn = repmat(kn, 1, d);
       end
       obj.muSigmaDist = NormInvGammaDist('mu', mn, 'k', kn, 'a', an, 'b', bn);
     end
  
     function m = marginal(obj)
       a = obj.muSigmaDist.a; b = obj.muSigmaDist.b; m = obj.muSigmaDist.mu; k = obj.muSigmaDist.k;
       m = StudentDist(2*a, m, b.*(1+k)./a); 
     end
     
  end
  
  methods(Static = true)
    function test
      prior = NormInvGammaDist('mu', 0, 'k', 0.01, 'a', 0.01, 'b', 0.01);
      p = Gauss_NormInvGammaDist(prior);
      x = rand(100,1);
      p = fit(p, 'data', x);
      pp = marginal(p);
      pp = var(p);
    end
  end
end