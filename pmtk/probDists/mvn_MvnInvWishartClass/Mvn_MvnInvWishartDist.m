classdef Mvn_MvnInvWishartDist < ParamDist 
% p(X|alpha) = int_m int_S N(X|m,S) NIW(m,S|alpha) 
% NIW(m,S|alpha) = N(m|mu, 1/k * S) IW(S| dof, Sigma)
% where alpha = (mu, Sigma, dof, k)
% Note that mu, Sigma are parameters of NIW, not parameters of Mvn
  properties
    muSigmaDist;
  end
  
  
  properties(SetAccess = 'private')
     ndims;
  end
  
  
  %% main methods
  methods
    function model = Mvn_MvnInvWishartDist(prior)
      % MvnMvnInvWishartDist(prior) where prior is of type MnvInvWishartDist 
      model.muSigmaDist = prior;
      model.ndims  = length(prior.mu);
    end
    
   
    function d = ndimensions(m)
      d= m.ndims;
    end
     
    function obj = fit(obj,varargin)
      % Update hyper-parameters
      % INPUT:
      %
      % 'data'     -        data(i,:) is case i
      % 'suffStat'
      [X,SS] = process_options(varargin,...
        'data'              ,[]         ,...
        'suffStat'          ,[]);
      if isempty(SS), SS = MvnDist.mkSuffStat(X); end
      %k0 = obj.k; m0 = obj.mu; S0 = obj.Sigma; v0 = obj.dof;
      k0 = obj.muSigmaDist.k; m0 = obj.muSigmaDist.mu;
      S0 = obj.muSigmaDist.Sigma; v0 = obj.muSigmaDist.dof;
      n = SS.n;
      kn = k0 + n;
      vn = v0 + n;
      Sn = S0 + n*SS.XX + (k0*n)/(k0+n)*(SS.xbar-m0)*(SS.xbar-m0)';
      mn = (k0*m0 + n*SS.xbar)/kn;
      %obj.mu = mn; obj.Sigma = Sn; obj.dof = vn; obj.k = kn;
      obj.muSigmaDist = MvnInvWishartDist('mu', mn, 'Sigma', Sn, 'dof', vn, 'k', kn);
    end
   
    function p = paramDist(obj)
      % Return current distribution over parameters, p(mu,Sigma) = NIW()
      %p = MvnInvWishartDist('mu', obj.mu, 'Sigma', obj.Sigma, 'dof', obj.dof, 'k', obj.k);
      p = obj.muSigmaDist;
    end
         
  end % methods

end

  