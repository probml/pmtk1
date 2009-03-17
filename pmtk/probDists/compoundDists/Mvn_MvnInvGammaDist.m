classdef Mvn_MvnInvGammaDist < CompoundDist
  % p(X,mu,sigma2|m,k,a,b) = N(X|mu, sigma2) NIG(mu,sigma2| m,k,a,b)
  
  properties
    muSigmaDist;
  end
  
  %% Main methods
  methods 
     function m = Mvn_MvnInvGammaDist(prior)
        if(nargin < 1), prior = []; end
        m.muSigmaDist = prior;
     end

    function obj = mode(model,covtype)
			if(nargin == 1)
				covtype = 'spherical';
			end
      obj.mu = model.muSigmaDist.mu;
      d = length(obj.mu);
			% We need to consider the case of diagonal vs spherical covariance
			switch lower(covtype)
				case 'spherical'
					obj.Sigma = model.muSigmaDist.b / (model.muSigmaDist.a + 1/2*(d + 2));
				case 'diagonal'
					obj.Sigma = model.muSigmaDist.b / (model.muSigmaDist.a + 1/2*(1 + 2));
			end
    end
     
     function obj = fit(obj, varargin)
       [X, SS,covtype] = process_options(...
         varargin, 'data', [], 'suffStat', [], 'covtype', 'spherical');
			 if isempty(SS), SS = MvnDist.mkSuffStat(X); end
       xbar = SS.xbar;
       if SS.n==0, return; end
			 n = SS.n;
       [n d] = size(X);
       m0 = obj.muSigmaDist.mu;
       k0 = obj.muSigmaDist.Sigma;
       a0 = obj.muSigmaDist.a;
       b0 = obj.muSigmaDist.b;
       kn = k0 + n; 
       mn = (k0*m0 + n*xbar)/kn;
			 switch lower(covtype)
				 case 'spherical'
				 	 an = a0 + n*d/2;
					 bn = b0 + 1/2*sum(diag( n*SS.XX + (k0*n)/(k0+n)*(SS.xbar-colvec(m0))*(SS.xbar-colvec(m0))' ));
				 case 'diagonal'
				 	 an = a0 + n/2;
					 bn = diag( b0*eye(d) + 1/2*(n*SS.XX + (k0*n)/(k0+n)*(SS.xbar-colvec(m0))*(SS.xbar-colvec(m0))'));
			 end

       obj.muSigmaDist = MvnInvGammaDist('mu', mn, 'Sigma', kn, 'a', an, 'b', bn);
     end
  
     function m = marginal(obj)
       a = obj.muSigmaDist.a; b = obj.muSigmaDist.b; m = obj.muSigmaDist.mu; k = obj.muSigmaDist.k;
       m = StudentDist(2*a, m, b.*(1+k)./a); 
     end
     
  end
  
end
