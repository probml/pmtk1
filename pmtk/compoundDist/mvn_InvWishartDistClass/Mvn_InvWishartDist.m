classdef Mvn_InvWishartDist < ParamDist 
% p(X|mu,Sigma_Sigma, Sigma_dof) = int_S N(X|mu,S) IW(S|Sigma,dof) 
  properties
    mu;
    SigmaDist;
  end
  
 
  properties(SetAccess = 'private')
     ndims;
  end
  
  
  %% main methods
  methods
    function model = Mvn_InvWishartDist(mu, SigmaPrior)
      % Mvn_InvWishartDist(mu, SigmaPrior) where SigmaPrior is of type InvWishartDist 
      model.mu = mu;
      model.SigmaDist = SigmaPrior;
      model.ndims  = length(mu);
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
      mu = obj.mu; 
      n = SS.n;
      T0 = obj.SigmaDist.Sigma;
      v0 = obj.SigmaDist.dof;
      vn = v0 + n;
      Tn = T0 + n*SS.XX +  n*(SS.xbar-mu)*(SS.xbar-mu)';
      obj.SigmaDist = InvWishartDist(vn, Tn);
    end
   
    function p = paramDist(obj)
      % Return current distribution over parameters
      p = ProductDist({ConstDist(obj.mu), obj.SigmaDist}, {'mu', 'Sigma'});
    end
    
  end % methods

end

  