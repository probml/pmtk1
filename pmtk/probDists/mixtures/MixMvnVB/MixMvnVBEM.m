classdef MixMvnVBEM < ProbDist
 
  % Variational Bayes for a mixture of multivariate normals
  
  properties
    nrestarts;
    % we do not store distributions or mixingDistrib.
    % Instead, we store posterior over their parameters, represented
    % as hyper params - see Bishop ch 10
    % The parameter defining the Dirichlet for the mixingDistrib
    alpha;
    % The following are the parameters for a MvnInvWishartDist() / MvnInvGammaDist().
    % We adopt the following notational convention
    % For MvnInvWishartDist() / MvnInvGammaDist()
    % p(mu | Sigma) \propto exp{-1/2*(x-mu)'k*Sigma^{-1}(x-mu)}
    % For MvnInvWishartDist()
    % p(Sigma) \propto |T|^{-(dof+d+1)/2} exp{-1/2*trace(Sigma*T^{-1})}
    % For MvnInvGammaDist()
    % p(sigma_j) \propto (1/sigma_j)^{-(dof+1)} exp{-(T(j,j) / sigma_j)}
    % Note that this choice of definition does not agree (in variable name) with
    % that given in MvnInvGammaDist(), but allows us to more compactly write code
    mu;
    T;
    dof;
    k;
    covtype; %'full', or 'diagonal', or 'spherical'

  end

  methods

    function model = MixMvnVBEM(varargin)
    [model.alpha, model.mu, model.k, model.T, model.dof, model.covtype, model.nrestarts] = processArgs(varargin, ...
      '-alpha', [], ...
      '-mu', [], ...
      '-k', [], ...
      '-T', [], ...
      '-dof', [], ...
      '-covtype', [], ...
      '-nrestarts', 1);
    end

    function model = fit(model, data, varargin)
      %error('not yet implemented')
      K = numel(model.alpha);
      %prior = cell(1,K);
      %for k=1:K
      %  prior{k} = mkPrior(MvnDist('-mu', model.mu(k,:)', '-Sigma', model.Sigma(:,:,k), '-covtype', model.covtype), '-data', data);
      %end
      nrestarts = model.nrestarts;
      param = cell(nrestarts, 1);
      L = zeros(nrestarts, 1);
      if(any(~strcmpi(model.covtype, 'full')))
        warning('Restricted covariances not rigourously tested yet');
      end
      for r=1:nrestarts
        [param{r}, L(r)] = VBforMixMvn(model.alpha, model.mu, model.k, model.T, model.dof, model.covtype, data, varargin{:});
      end
      best = argmax(L);
      model.alpha = param{best}.alpha;
      model.mu = param{best}.mu;
      model.T = param{best}.T;
      model.dof = param{best}.dof;
      model.k = param{best}.k;
    end
    
    function [ph, LL] = conditional(model,data)
      % ph(i,k) = (1/S) sum_s p(H=k | data(i,:),params(s)), a DiscreteDist
      % This is the posterior responsibility of component k for data i
      % LL(i) = log p(data(i,:) | params)  is the log normalization const
    end

     function logp = logprob(model,data)
     end
    

  end % methods

end