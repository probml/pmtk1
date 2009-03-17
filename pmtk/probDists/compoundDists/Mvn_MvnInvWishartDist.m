classdef Mvn_MvnInvWishartDist < CompoundDist 
% p(X,m,s|alpha) = N(X|m,S) NIW(m,S|alpha) 
% where NIW(m,S|alpha) = N(m|mu, 1/k * S) IW(S| dof, Sigma)
% and alpha = (mu, Sigma, dof, k)
  properties
    muSigmaDist;
  end
  
  
  
  %% main methods
  methods
    function model = Mvn_MvnInvWishartDist(prior)
      if(nargin < 1),prior = [];end
      % Mvn_MvnInvWishartDist(prior) where prior is of type MvnInvWishartDist 
      model.muSigmaDist = prior;
      %model.ndims  = length(prior.mu);
    end
    
   
    function d = ndimensions(m)
      d= length(m.muSigmaDist.mu); % m.ndims;
    end
     
    function obj = mode(model)
      obj.mu = model.muSigmaDist.mu;
      d = length(obj.mu);
      obj.Sigma = model.muSigmaDist.Sigma / (model.muSigmaDist.dof + d + 2);
    end
    
    function pp = marginal(model)
      % integrate out mu and Sigma
      muSigmaDist = model.muSigmaDist;
      mu = muSigmaDist.mu; T = muSigmaDist.Sigma; dof = muSigmaDist.dof; k = muSigmaDist.k;
      d = ndimensions(model);
      pp = MvtDist(dof - d + 1, mu, T*(k+1)/(k*(dof-d+1)));
      assert(isposdef(pp.Sigma))
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
      if isempty(SS), SS = MvnDist().mkSuffStat(X); end
      %k0 = obj.k; m0 = obj.mu; S0 = obj.Sigma; v0 = obj.dof;
      k0 = obj.muSigmaDist.k; m0 = obj.muSigmaDist.mu;
      S0 = obj.muSigmaDist.Sigma; v0 = obj.muSigmaDist.dof;
      if SS.n==0, return; end
      n = SS.n;
      kn = k0 + n;
      vn = v0 + n;
      Sn = S0 + n*SS.XX + (k0*n)/(k0+n)*(SS.xbar-colvec(m0))*(SS.xbar-colvec(m0))';
      mn = (k0*colvec(m0) + n*SS.xbar)/kn;
      %obj.mu = mn; obj.Sigma = Sn; obj.dof = vn; obj.k = kn;
      obj.muSigmaDist = MvnInvWishartDist('mu', mn, 'Sigma', Sn, 'dof', vn, 'k', kn);
    %assert(~isnan(mn))
    end
   
  
    
  end % methods

end

  
