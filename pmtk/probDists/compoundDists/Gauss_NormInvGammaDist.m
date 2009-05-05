classdef Gauss_NormInvGammaDist < ProbDist
  % p(X,mu,sigma2|m,k,a,b) = N(X|mu, sigma2) NIG(mu,sigma2| m,k,a,b)
  
  properties
    muSigmaDist;
    productDist;
  end
  
  %% Main methods
  methods 
     function m = Gauss_NormInvGammaDist(varargin)
        [m.muSigmaDist, m.productDist] = processArgs(varargin, ...
          '-prior', [], '-productDist', false);
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
       obj.muSigmaDist = NormInvGammaDist('-mu', mn, '-k', kn, '-a', an, '-b', bn);
     end
  
     function p = logprob(obj, X)
       % p(i) = log p(X(i)) log marginal likelihood
       p = logprob(marginalizeOutParams(obj), X);
     end
   
     function m = marginalizeOutParams(obj)
       a = obj.muSigmaDist.a; b = obj.muSigmaDist.b;
       mu = obj.muSigmaDist.mu; k = obj.muSigmaDist.k;
       m = StudentDist(2*a, mu, b.*(1+k)./a, obj.productDist); 
     end
   
     
  end
  
end