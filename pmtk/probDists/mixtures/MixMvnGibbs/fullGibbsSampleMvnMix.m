function [muS, SigmaS, mixS, latentS] = latentGibbsSampleMvnMix(distributions, mixingWeights, data, varargin)
  [Nsamples, Nburnin, thin, verbose] = processArgs(varargin, ...
    '-Nsamples'  , 1000, ...
    '-Nburnin'   , 500, ...
    '-thin'      , 1, ...
    '-verbose'   , false );
  
  % Initialize the model
  K = numel(distributions);
  [nObs,d] = size(data);
  prior = cell(K,1);
  Sigma = zeros(d,d,K);
  mu0 = zeros(d,K); v0 = zeros(K,1); k0 = zeros(K,1); S0 = zeros(d,d,K);
  %[mu, assign] = kmeansSimple(data, K);
  for k=1:K
    %C = cov(data(assign == k,:));
    %Sigma(:,:,k) = C + 0.01*diag(diag(C));
    if(isa(distributions{k}.prior, 'char'))
      distributions{k} = initPrior(distributions{k}, data);
    end
    prior{k} = distributions{k}.prior;
    covtype{k} = distributions{k}.covtype;
    mu0(:,k) = distributions{k}.prior.mu;
    switch class(prior{k})
      case 'MvnInvWishartDist'
        k0(k) = distributions{k}.prior.k;
        v0(k) = distributions{k}.prior.dof;
        S0(:,:,k) = distributions{k}.prior.Sigma;
      case 'MvnInvGammaDist'
        k0(k) = distributions{k}.prior.Sigma;
        if(any(mean(distributions{k}.prior.a) ~= distributions{k}.prior.a))
          error('gibbs:MvnInvGammaDist:dofError', 'Require dof for each gamma prior to be equal within a cluster for conjugate MvnInvGamma prior');
        end
        v0(k) = distributions{k}.prior.a(1); % We assume (rather, require) shared degrees of freedom across the inverse gamma priors
        S0(:,:,k) = diag(distributions{k}.prior.b);
    end
    [mu(:,k), Sigma(:,:,k)] = sample(prior{k});
  end

  % It will be convenient to work with mu as a row vector in this code
  mu = mu'; mu0 = mu0';
  
  alpha = mixingWeights.prior.alpha;
  mixWeights = mixingWeights.T;
  post = cell(K,1);
  if(verbose), fprintf('Full Gibbs Sampling initiated.  Starting to collect samples\n'); end;
  
  nSamples = ceil((Nsamples - Nburnin) / thin);
  latentsamples = zeros(nObs,nSamples);
  musamples = zeros(d,nSamples,K);
  Sigmasamples = zeros(d,d,nSamples,K);
  mixsamples = zeros(K,nSamples);

  keep = 1;
  logRik = zeros(nObs,K);
  % Get samples
  for itr=1:Nsamples
    if(mod(itr,500) == 0 && verbose), fprintf('Collected %d samples \n', itr); end;
    % sample the latent variables conditional on the parameters
    for k=1:K
      XC = bsxfun(@minus, data, mu(k,:));
      logRik(:,k) = log(mixWeights(k) + sqrt(eps)) - 1/2*logdet(2*pi*Sigma(:,:,k)) - 1/2*sum((XC*inv(Sigma(:,:,k))).*XC,2);
    end
    pred = exp(normalizeLogspace(logRik));
    latent = zeros(nObs, 1);
    for n=1:nObs
      latent(n) = sample(pred(n,:),1);
    end
    nclust = histc(latent, 1:K);
    for k=1:K
      if(nclust(k) > 0)
        dataclust = data(latent == k,:);
        xbar = rowvec(mean(dataclust));
        sampledMu = (nclust(k)*xbar + k0(k)*mu0(k,:)) / (nclust(k) + k0(k));
        switch lower(covtype{k})
          case 'full'
            Sn = (S0(:,:,k) + nclust(k)*cov(dataclust,1) + k0(k)*nclust(k) / (k0(k) + nclust(k))*(xbar - mu0(k,:))*(xbar - mu0(k,:))');
            Sigma(:,:,k) = iwishrnd(Sn, v0(k) + nclust(k));
          case 'diagonal'
            Sn = (S0(:,:,k) + 1/2*diag(diag(nclust(k)*cov(dataclust,1) + k0(k)*nclust(k) / (k0(k) + nclust(k))*(xbar - mu0(k,:))*(xbar - mu0(k,:))')));
            Sigma(:,:,k) = diag(invchi2rnd(2*(v0(k) + nclust(k)/2), diag(Sn) ./ (v0(k) + nclust(k)/2)));
          case 'spherical'
            Sn = (S0(:,:,k) + 1/2*sum(diag(nclust(k)*cov(dataclust,1) + k0(k)*nclust(k) / (k0(k) + nclust(k))*(xbar - mu0(k,:))*(xbar - mu0(k,:))'))*eye(d));
            Sigma(:,:,k) = invchi2rnd(2*(v0(k) + nclust(k)*d/2), Sn(1,1) ./ (v0(k) + nclust(k)*d/2)) * eye(d); % first element suffices
        end
      else
        xbar = mu0(k,:);
        switch lower(covtype{k})
          case 'full'
            Sigma(:,:,k) = iwishrnd(S0(:,:,k), v0(k));
          case 'diagonal'
            Sigma(:,:,k) = diag(invchi2rnd(2*v0(k), diag(S0(:,:,k))./v0(k)));
          case 'spherical'
            Sigma(:,:,k) = invchi2rnd(2*v0(k), S0(1,1,k)./v0(k)) * eye(d); % first element suffices
        end
      end
      mu(k,:) = mvnrnd(sampledMu, Sigma(:,:,k) / (k0(k) + nclust(k)));
    end % of k=1:K
    % Conditional on the sampled assignments, fit and sample from the Dirichlet-Multinomial that defines the mixing weights
    mixWeights = normalize(nclust + alpha);

    % Store the results if we are past burnin and if thinning permits
    if(itr > Nburnin && mod(itr, thin)==0)
      latentsamples(:,keep) = colvec(latent);
      mixsamples(:,keep) = colvec(mixWeights);
      for k=1:K
        musamples(:,keep,k) = colvec(mu(k,:));
        Sigmasamples(:,:,keep,k) = Sigma(:,:,k);
      end
      keep = keep + 1;
    end
  end % of itr=1:Nsamples
  latentS = SampleDist(latentsamples);
  mixS = SampleDist(mixsamples);
  muS = cell(K,1); SigmaS = cell(K,1);
  for k=1:K
    muS{k} = SampleDist(musamples(:,:,k));
    SigmaS{k} = SampleDist(Sigmasamples(:,:,k));
  end

end