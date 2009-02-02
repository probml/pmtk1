classdef Mvn_InvWishartDist < CompoundDist 
% p(X, Sigma | mu,Sigma_Sigma, Sigma_dof) = N(X|mu,Sigma) IW(Sigma|Sigma_Sigma,Sigma_dof) 
  properties
    mu;
    SigmaDist;
  end
  

  
  %% main methods
  methods
    function model = Mvn_InvWishartDist(mu, SigmaPrior)
      if(nargin ==0),mu = []; SigmaPrior = [];end
      % Mvn_InvWishartDist(mu, SigmaPrior) where SigmaPrior is of type InvWishartDist 
      model.mu = mu;
      model.SigmaDist = SigmaPrior;
    end
   
    function d = ndimensions(m)
      d= length(m.mu); 
    end
     
    function pp = marginal(model)
      % integrate out Sigma
      T = model.SigmaDist.Sigma; dof = model.SigmaDist.dof; 
      d = ndimensions(model);
      % Same as the MNVIW result, except the last term is missing a factor
      % of (k+1)/k
      pp = MvtDist(dof - d + 1, model.mu, T/(dof-d+1));
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
      mu = obj.mu; 
      n = SS.n;
      T0 = obj.SigmaDist.Sigma;
      v0 = obj.SigmaDist.dof;
      vn = v0 + n;
      Tn = T0 + n*SS.XX +  n*(SS.xbar-mu)*(SS.xbar-mu)';
      obj.SigmaDist = InvWishartDist(vn, Tn);
    end
   
  
    
  end % methods

end

  