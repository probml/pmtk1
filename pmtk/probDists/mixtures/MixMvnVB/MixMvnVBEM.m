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

    function [mixingDistrib, distributions] = convertToDist(model)
      % This function returns as dists the posterior distributions
      mixingDistrib = DirichletDist(colvec(model.alpha));
      K = numel(model.alpha); d = size(model.mu,2);
      distributions = cell(K,1);
      for k=1:K
        switch lower(model.covtype{k})
          case 'full'
            distributions{k} = MvnInvWishartDist('mu', model.mu(k,:)', 'Sigma', model.T(:,:,k), 'dof', model.dof(k), 'k', model.k(k));
          case {'diagonal', 'spherical'}
            distributions{k} = MvnInvGammaDist('mu', model.mu(k,:)', 'Sigma', model.k(k), 'a', model.dof(k)*ones(1,d), 'b', rowvec(diag(model.T(:,:,k))));
        end
      end
    end

    function [mixingDistrib, marginalDist] = marginal(model)
      % Same are convertToDist, but instead returns the marginal distributibutions
      % in place of the posterior distributions.
      % This is usefule for the conditional() and logprob() functions
      [mixingDistrib, distributions] = convertToDist(model);
      K = numel(distributions);
      conjugateDist = cell(K,1); marginalDist = cell(K,1);
      for k=1:K
        % Marginalize out the parameters
        switch class(distributions{k})
          case 'MvnInvWishartDist'
            conjugateDist{k} = Mvn_MvnInvWishartDist(distributions{k});
          case 'MvnInvGammaDist'
            conjugateDist{k} = Mvn_MvnInvGammaDist(distributions{k});
        end %switch
      marginalDist{k} = marginal(conjugateDist{k});
      end
      %mixMarg = MixModel('-distributions', marginalDist, '-mixingDistrib', mixingDistrib);
    end
    
%    function [ph, LL] = conditional(model,data)
    function ph = conditional(model,data)
      % ph(i,k) = (1/S) sum_s p(H=k | data(i,:),params(s)), a DiscreteDist
      % This is the posterior responsibility of component k for data i
      % LL(i) = log p(data(i,:) | params)  is the log normalization const
      [mixingDistrib, marginalDist] = marginal(model);
      K = numel(marginalDist);
      T = zeros([size(data,1), K]);
      for k=1:K
        T(:,k) = logprob(marginalDist{k}, data);
      end % for
      ph = DiscreteDist(exp(normalizeLogspace(T))');
    end

    function logp = logprob(model,data)
      % logp(i) = log int_{params} p(data(i,:), params)
      %  = log sum_k int_{params}p(data(i,:), h=k, params)
      [mixingDistrib, marginalDist] = marginal(model);
      mixWeights = pmf(DiscreteDist(normalize(colvec(mixingDistrib.alpha))));
      [n,d] = size(data); K = numel(marginalDist);
      logp = zeros(n,K);
      for k=1:K
        logp(:,k) = log(mixWeights(k)+eps) + logprob(marginalDist{k},data);
      end
        logp = logsumexp(logp,2);
    end
    

  end % methods

end