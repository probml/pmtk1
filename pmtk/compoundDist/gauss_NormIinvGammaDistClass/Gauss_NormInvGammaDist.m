classdef Gauss_NormInvGammaDist < ParamDist
  
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
       xbar = mean(X); n = size(X,1);
       m0 = obj.muSigmaDist.mu;
       k0 = obj.muSigmaDist.k;
       a0 = obj.muSigmaDist.a;
       b0 = obj.muSigmaDist.b;
       kn = k0 + n;
       mn = (k0*m0 + n*xbar)/kn;
       an = a0 + n/2;
       bn = b0 + 0.5*sum((X-xbar).^2) + 0.5*n*k0*(m0-xbar)^2/(k0+n);
       obj.muSigmaDist = NormInvGammaDist('mu', mn, 'k', kn, 'a', an, 'b', bn);
     end
  
  end
  
end