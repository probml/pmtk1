classdef Mvn_MvnDist < ParamDist 
% p(X|mu,mu_Sigma,Sigma) = int_m N(X|m,Sigma) N(m|mu,mu_Sigma) 
  properties
    muDist;
    Sigma;
  end
  
  properties(SetAccess = 'private')
     ndims;
  end
  
  
  %% main methods
  methods
    function model = Mvn_MvnDist(muPrior, Sigma)
      % MvnMvnDist(muPrior, Sigma) where muPrior is of type MnvDist 
      model.muDist = muPrior;
      model.Sigma = Sigma;
      model.ndims  = length(muPrior.mu);
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
      S0 = obj.muDist.Sigma; S0inv = inv(S0);
      mu0 = obj.muDist.mu;
      S = obj.Sigma; Sinv = inv(S);
      n = SS.n;
      Sn = inv(inv(S0) + n*Sinv);
      obj.muDist = MvnDist(Sn*(n*Sinv*SS.xbar + S0inv*mu0), Sn);
    end
   
    function p = paramDist(obj)
      % Return current distribution over parameters
      p = ProductDist({obj.muDist, ConstDist(obj.Sigma)}, {'mu', 'Sigma'});
    end
    
  end % methods

end

  