classdef MixMvnGibbs < MixMvn

  properties
    samples;
  end
 
  % Gibbs sampling for a mixture of multivariate normals
  
  methods
     function model = MixMvnGibbs(varargin)
       if nargin == 0; return; end
      [nmixtures,mixingWeights,distributions,model.transformer,model.verbose] = process_options(varargin,...
        'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[],'verbose',true);
      if(~isempty(distributions)),nmixtures = numel(distributions);end

      if(isempty(mixingWeights) && ~isempty(nmixtures))
        mixingWeights = DiscreteDist('-T',normalize(ones(nmixtures,1)), '-prior', DirichletDist(ones(nmixtures,1)) );
      end
      model.mixingDistrib = mixingWeights;
      if(isempty(distributions)&&~isempty(model.mixingDistrib))
        distributions = copy(MvnDist(),nstates(model.mixingDistrib),1);
      end
      model.distributions = distributions;
    end
    
    function ph = inferLatent(model,X)
      % ph(i,k) = p(H=k | data(i,:),params) a DiscreteDist
      % This is the posterior responsibility of component k for data i
      % LL(i) = log p(data(i,:) | params)  is the log normalization constant
      if(isempty(model.samples)), error('MixMvnGibbs', 'You must call fit first'); end;
      K = numel(model.distributions);
      S = size(model.samples.mu{1}.samples,2);
      [n,d] = size(X);
      logRik = zeros(n,K,S);
      Sigma = model.samples.Sigma;
      mu = model.samples.mu;
      mixW = model.samples.mixingWeights;
      for s=1:S
        for k=1:K
          XC = bsxfun(@minus, X, mu{k}.samples(:,s)');
          logRik(:,k,s) = log(mixW.samples(k,s)) - 1/2*logdet(2*pi*Sigma{k}.samples(:,:,s)) - 1/2*sum((XC*inv(Sigma{k}.samples(:,:,s))).*XC,2);
        end
      end
      logRik = mean(logRik,3);
      [Rik, LL] = normalizeLogspace(logRik);
      Rik = exp(Rik);
      ph = DiscreteDist('-T', Rik');
    end

    function x = sample(model, n)
      if(isempty(model.samples)), error('MixMvnGibbs','You must call fit first'); end;
      d = numel(model.distributions{1}.mu);
      k = sample(model.mixingDistrib,n);
      mu = zeros(d,n); Sigma = zeros(d,d,n);
      for i=1:n
        mu = colvec(sample(model.samples.mu{k(i)}, 1));
        Sigma = sample(model.samples.Sigma{k(i)}, 1);
        x(i,:) = sample(MvnDist(mu,Sigma), 1);
      end
    end

    function l = logprob(model, X)
      if(isempty(model.samples)), error('MixMvnGibbs', 'You must call fit first'); end;
      K = numel(model.distributions);
      S = size(model.samples.mu{1}.samples,2);
      [n,d] = size(X);
      l = zeros(n,K,S);
      Sigma = model.samples.Sigma;
      mu = model.samples.mu;
      mixW = model.samples.mixingWeights;
      for s=1:S
        for k=1:K
          XC = bsxfun(@minus, X, mu{k}.samples(:,s)');
          l(:,k,s) = log(mixW.samples(k,s)) - 1/2*logdet(2*pi*Sigma{k}.samples(:,:,s)) - 1/2*sum((XC*inv(Sigma{k}.samples(:,:,s))).*XC,2);
        end
      end
      l = mean(l,3);
      l = logsumexp(l,2);
    end


% We don't want this
%{
    function model = mean(model)
      sampleDists = model.samples;
      % Performs full Bayes Model averaging by averaging over each sampled mu, Sigma, and mixing weights
      mu = sampleDists.mu;
      Sigmatmp = sampleDists.Sigma;
      mix = sampleDists.mixingWeights;
      [N, d, K] = size(mu);
      Sigma = recoversigma(model);
      % perform model averaging
      SigmaAvg = mean(Sigma,3);
      mixAvg = mean(sampleDists.mixingWeights);
      for k=1:K
        model.distributions{k}.mu = colvec(mean(sampleDists.mu{k}));
        model.distributions{k}.Sigma = SigmaAvg(:,:,:,k);
      end
      model.mixingDistrib.T = colvec(mixAvg);
    end
%}

    function [model, latent] = fit(model,X,varargin)
      [method, fixlatent, Nsamples, Nburnin, thin, verbose] = processArgs(varargin, ...
        '-method', 'collapsed', ...
        '-fixlatent', 'false', ...
        '-Nsamples', 1000, ...
        '-Nburnin', 500, ...
        '-thin', 1, ...
        '-verbose', true);
      switch lower(method)
        case 'full'
          [muS, sigmaS, mixS, latent] = fullGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, Nsamples, Nburnin, thin, verbose);
        case 'collapsed'
          [muS, sigmaS, mixS, latent] = collapsedGibbsSampleMvnMix(model.distributions, model.mixingDistrib, X, Nsamples, Nburnin, thin, verbose);
      end

      if(fixlatent)
        [muS, sigmaS, mixS, latent] = processLabelSwitching(muS, sigmaS, mixS, latent, X);
      end
      model.samples.mu = muS;
      model.samples.Sigma = sigmaS;
      model.samples.mixingWeights = mixS;
      K = numel(muS);
      for k=1:K
        model.distributions{k}.mu = colvec(mean(muS{k}));
        model.distributions{k}.Sigma = mean(sigmaS{k});
      end
      model.mixingDistrib.T = colvec(mean(mixS));
    end

    function [] = traceplot(model)
      if(isempty(model.samples)), error('MixMvnGibbs:traceplot:notFit', 'Must first fit Mixture to data'); end;
      mu = model.samples.mu;
      K = numel(mu);
      figure(); hold on;
      [nRows, nCols] = nsubplots(K);
      for k=1:K
        subplot(nRows, nCols, k);
        plot(mu{k}.samples);
        xlabel(sprintf('Distribution %d', k));
      end
    end

    function [] =  convergencePlot(model,X)
      if(isempty(model.samples)), error('MixMvnGibbs:traceplot:notFit', 'Must first fit Mixture to data'); end;
      mu = model.samples.mu;
      Sigma = model.samples.Sigma;
      mixW = model.samples.mixingWeights;
      S = size(model.samples.mu{1}.samples,2);
      K = numel(model.distributions);
      lognormconst = zeros(S,1); logp = zeros(S,1);
      for s=1:S
        for k=1:K
          XC = bsxfun(@minus, X, mu{k}.samples(:,s)');
          l(:,k,s) = log(mixW.samples(k,s)) - 1/2*logdet(2*pi*Sigma{k}.samples(:,:,s)) - 1/2*sum((XC*inv(Sigma{k}.samples(:,:,s))).*XC,2);
        end
      lognormconst(s) = logsumexp(logsumexp(l(:,:,s),2));
      end
      logp = squeeze(logsumexp(sum(l,1),2));
      figure(); hold on;
      subplot(2,1,1); plot(logp); title('Log probability of data');
      subplot(2,1,2); plot(lognormconst); title('Log normalization constant');
    end

    function [xrange] = plotRange(model)
      K = numel(model.distributions);
      d = ndimensions(model.distributions{1});
      switch d
        case 1
          xrange = zeros(K,2);
          for k=1:K
            xrange(k,:) = plotRange(model.distributions{k});
          end
          xrange = [min(xrange(:,1)), max(xrange(:,2))];
        case 2
          for k=1:K
            xrange(k,:) = plotRange(model.distributions{k});
          end
          xrange = [min(xrange(:,1)), max(xrange(:,2)), min(xrange(:,3)), max(xrange(:,4))];
      end
    end
  
  end % methods
  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% special purpose code after this point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [muS, SigmaS, mixS, latentS] = fullGibbsSampleMvnMix(distributions, mixingWeights, data, varargin)
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
        xbar = mu0(k,:); sampledMu = xbar;
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
    SigmaS{k} = SampleDist(Sigmasamples(:,:,:,k));
  end

end

function [muS, sigmaS, mixS, latentS] = collapsedGibbsSampleMvnMix(distributions, mixingWeights, data, varargin)
  % Collapsed Gibbs sampling for a mixture of MVNs
  [Nsamples, Nburnin, thin, verbose] = processArgs(varargin, ...
    '-Nsamples'  , 1000, ...
    '-Nburnin'   , 500, ...
    '-thin'    , 1, ...
    '-verbose'   , true );
  [nObs,d] = size(data); K = numel(distributions);
  nSamples = ceil((Nsamples - Nburnin) / thin); keep = 1;
  latent = zeros(nSamples, nObs);
  alpha = mixingWeights.prior.alpha;
  if(verbose), fprintf('Collapsed Gibbs Sampling initiated.  Starting to collect samples\n'); end;
  
  latent(1,:) = unidrnd(K,1,nObs);
  SSn = zeros(K,1); SSxbar = zeros(d,K); SSXX = zeros(d,d,K); SSXX2 = zeros(d,d,K);
  covtype = cell(1,K);

  % since each MVN could have a different covariance structure, we need to do this.
  for k=1:K
    if(isa(distributions{k}.prior,'char'))
      distributions{k} = initPrior(distributions{k},data);
    end
    covtype{k} = distributions{k}.covtype;
    SS = MvnDist().mkSuffStat(data(latent(1,:)' == k,:));
    SSn(k) = SS.n;
    SSxbar(:,k) = SS.xbar;
    SSXX(:,:,k) = SS.XX;
    SSXX2(:,:,k) = SS.XX2;
    mup(:,k) = distributions{k}.prior.mu;
    switch lower(covtype{k})
      case 'full'
        kp(k) = distributions{k}.prior.k;
        vp(k) = distributions{k}.prior.dof;
        Sp(:,:,k) = distributions{k}.prior.Sigma;
      case 'diagonal'
        kp(k) = distributions{k}.prior.Sigma;
        if(any(distributions{k}.prior.a ~= distributions{k}.prior.a(1)))
          error('gibbs:diagonal:dofError', 'Require dof for each gamma prior to be equal within a cluster for conjugate MvnInvGamma prior');
        end
        vp(k) = distributions{k}.prior.a(1); % We assume (rather, require) shared degrees of freedom across the inverse gamma priors
        Sp(:,:,k) = diag(distributions{k}.prior.b);
      case 'spherical'
        kp(k) = distributions{k}.prior.Sigma;
        vp(k) = distributions{k}.prior.a;
        Sp(:,:,k) = distributions{k}.prior.b*eye(d);
    end
  end
  % Later we will need these in map estimation
  %kp = k0; vp = v0; Sp = S0; mup = mu0;

  curlatent = latent(1,:);


  % Collect samples
  if(verbose), fprintf('Samples collected: '); end;
  for itr=1:Nsamples
    if(verbose && mod(itr,100) == 0), fprintf('%d... ', itr); end;
    for obs=1:nObs
      prob = zeros(1,K);
      % For each iteration and observation, get the current assignment of the current observation
      % update the posterior parameters for its current cluster to reflect the absence of this observation
      kobs = curlatent(obs);
      xi = data(obs,:);
      [SSxbar, SSn, SSXX, SSXX2] = SSremoveXi(SSxbar, SSn, SSXX, SSXX2, kobs, xi);
      for k = 1:K
          [k0,v0,S0,m0] = updatePost(covtype{k}, kp(k), vp(k), mup(:,k), Sp(:,:,k), SSxbar(:,k), SSn(k), SSXX(:,:,k), SSXX2(:,:,k), xi);
          % In previous version of this code, we worked with objects that were alway of type
          % MvtDist or StudentDist, updated the parameters of these objects, and then called logProb.
          % This was slow, since logProb did a lot more than we needed to actually do.
          % The new version of the code computes the necessary marginal probability directly
          % Credit to Matt Dunham for the suggestion
          switch covtype{k}
            case 'full'
              vn = v0 + 1;
              Sn = S0 + k0/(k0+1)*(xi'- m0)*(xi'- m0)';
              logZ = (d/2)*(log(k0) - log(k0 + 1) - log(pi)) + mvtGammaln(d,vn/2) - mvtGammaln(d,v0/2);
              dist = (v0/2)*logdet(S0) - (vn/2)*logdet(Sn);
              logmprob = dist + logZ;
            case 'spherical'
              an = v0 + d/2;
              bn = S0 + 1/2*sum(diag( k0/(k0+1)*(xi'- m0)*(xi'- m0)' ))*eye(d);
              logZ = (d/2)*(log(k0) - log(k0+1) - log(2*pi)) + gammaln(an) - gammaln(v0);
              dist = v0*gammaln(diag(S0)) - an*gammaln(bn);
              logmprob = dist + logZ;
            case 'diagonal'
              an = v0 + 1/2;
              bn = S0 + 1/2*diag((k0/(k0+1)*(xi'- m0)*(xi'- m0)'));
              logZ = (d/2)*(log(k0) - log(k0+1) - log(2*pi)) + sum(gammaln(an) - gammaln(v0));
              dist = v0.*gammaln(diag(S0)) - an.*gammaln(bn);
              logmprob = dist + logZ;
          end
        lognij = log(SSn(k) + alpha(k));
        % Now find the probability of this observation being in each cluster
        prob(k) =  lognij + logmprob;
      end % for clust = 1:K
      % normalize, sample, and then update the sufficient statistics to reflect the new assignment
      prob = exp(normalizeLogspace(prob));
      knew = sample(prob,1);
      curlatent(obs) = knew;
      [SSxbar, SSn, SSXX, SSXX2] = SSaddXi(SSxbar, SSn, SSXX, SSXX2, curlatent(obs), data(obs,:));
    end % obs=1:n
    % Store the results if past burnin and if thinning permits
    if(itr > Nburnin && mod(itr, thin)==0)
      latent(keep,:) = curlatent;
      keep = keep + 1;
    end
  end
  if(verbose)
    fprintf('\n')
  end

  % With Gibbs sampling complete, create the mcmc struct needed for returning
  latentS = SampleDist(latent');
  musamples = zeros(d, nSamples, K);
  Sigmasamples = zeros(d, d, nSamples, K);
  mix = zeros(K, nSamples);
  for itr=1:nSamples
    counts = histc(latent(itr,:)', 1:K);
    mix(:,itr) = normalize(counts + alpha);
    nclust = histc(latent(itr,:)', 1:K);
    for k=1:K
      if(nclust(k) > 0)
        dataclust = data(latent(itr,:)' == k,:);
        xbar = colvec(mean(dataclust));
        musamples(:,itr,k) = (nclust(k)*xbar + kp(k)*mup(:,k)) / (nclust(k) + kp(k));
        switch lower(covtype{k})
          case 'full'
            Sigmasamples(:,:,itr,k) = (Sp(:,:,k) + nclust(k)*cov(dataclust,1) + kp(k)*nclust(k) / (kp(k) + nclust(k))*(xbar - mup(:,k))*(xbar - mup(:,k))') / (nclust(k) + vp(k) + d + 2);
          case 'diagonal'
            Sigmasamples(:,:,itr,k) = (Sp(:,:,k) + 1/2*diag(diag(nclust(k)*cov(dataclust,1) + kp(k)*nclust(k) / (kp(k) + nclust(k))*(xbar - mup(:,k))*(xbar - mup(:,k))'))) / (nclust(k)/2 + vp(k) + 1/2 + 2);
          case 'spherical'
            Sigmasamples(:,:,itr,k) = (S0(:,:,k) + 1/2*sum(diag(nclust(k)*cov(dataclust,1) + kp(k)*nclust(k) / (kp(k) + nclust(k))*(xbar - mup(:,k))*(xbar - mup(:,k))'))*eye(d)) / (nclust(k)*d/2 + vp(k) + d/2 + 2);
        end %switch
      else
        musamples(:,itr,k) = mup(:,k);
        switch lower(covtype{k})
          case 'full'
            Sigmasamples(:,:,itr,k) = Sp(:,:,k) / (vp(k) + d + 2);
          case 'diagonal'
            Sigmasamples(:,:,itr,k) = Sp(:,:,k) / (vp(k) + 1/2 + 2);
          case 'spherical'
            Sigmasamples(:,:,itr,k) = Sp(:,:,k) / (vp(k) + d/2 + 2);
        end
      end
    end % for k=1:K
  end % itr=1:nSamples
  muS = cell(K,1); sigmaS = cell(K,1);
  for k=1:K
    muS{k} = SampleDist(musamples(:,:,k));
    sigmaS{k} = SampleDist(Sigmasamples(:,:,:,k));
  end
  mixS = SampleDist(mix);
end
 
function [SSxbar, SSn, SSXX, SSXX2] = SSaddXi(SSxbar, SSn, SSXX, SSXX2, k, xi)
  % Add contribution of xi to SS for cluster k
  SSxbar(:,k) = (SSn(k)*SSxbar(:,k) + xi') / (SSn(k) + 1);
  SSXX2(:,:,k) = (SSn(k)*SSXX2(:,:,k) + xi'*xi) / (SSn(k) + 1);
  SSXX(:,:,k) = ( (SSn(k) + 1)*SSXX2(:,:,k) - (SSn(k) + 1)*SSxbar(:,k)*SSxbar(:,k)' ) / (SSn(k) + 1);
  SSn(k) = SSn(k) + 1;
end

function [SSxbar, SSn, SSXX, SSXX2] = SSremoveXi(SSxbar, SSn, SSXX, SSXX2, k, xi)
  if(SSn(k) == 1)
    SSxbar(:,k) = 0;
    SSXX2(:,:,k) = 0;
    SSXX(:,:,k) = 0;
    SSn(k) = 0;
  return;
  end
  SSxbar(:,k) = (SSn(k)*SSxbar(:,k) - xi') / (SSn(k) - 1);
  SSXX2(:,:,k) = (SSn(k)*SSXX2(:,:,k) - xi'*xi) / (SSn(k) - 1);
  SSXX(:,:,k) = ( (SSn(k) - 1)*SSXX2(:,:,k) - (SSn(k) - 1)*SSxbar(:,k)*SSxbar(:,k)' ) / (SSn(k) - 1);
  SSn(k) = SSn(k) - 1;
end

function [kn,vn,Sn,mn] = updatePost(covtype, k0, v0, m0, S0, xbar, SSn, SSXX, SSXX2, xi)
  if(SSn == 0)
    kn = k0; mn = m0; Sn = S0; vn = v0; return;
  end
  n = SSn; d = length(xi);
  kn = k0 + n;
  mn = (k0*m0 + n*xbar)/kn;
  switch lower(covtype)
    case 'full'
      vn = v0 + n;
      Sn = S0 + n*SSXX + k0*n/(k0 + n)*(xbar - m0)*(xbar - m0)';
    case 'diagonal'
      vn = v0 + n/2;
      Sn = S0 + 1/2*diag(diag(n*SSXX + k0*n/(k0 + n)*(xbar - m0)*(xbar - m0)'));
    case 'spherical'
      vn = v0 + n*d/2;
      Sn = (S0 + 1/2*sum(diag(n*SSXX + k0*n/(k0 + n)*(xbar - m0)*(xbar - m0)')))*eye(d);
  end
end

    function [muoutDist, SigmaoutDist, mixoutDist, latentoutDist] = processLabelSwitching(muDist, SigmaDist, mixDist, latentDist, X, varargin)
      % Implements the KL - algorithm for label switching from 
      %@article{ stephens2000dls,
      % title = "{Dealing with label switching in mixture models}",
      % author = "M. Stephens",
      % journal = "Journal of the Royal Statistical Society. Series B, Statistical Methodology",
      % pages = "795--809",
      % year = "2000",
      % publisher = "Blackwell Publishers"
      %}
      [verbose, maxitr] = processArgs(varargin, '-verbose', true, '-maxitr', inf);
      %muDist = dists.muDist;
      %SigmaDist = dists.SigmaDist;
      %mixDist = dists.mixDist;
      %latentDist = dists.latentDist;

      nSamples = size(latentDist.samples, 2);
      [nObs,d] = size(X);
      K = size(mixDist.samples,1);

      latent = latentDist.samples';
      mix = mixDist.samples';
      mu = zeros(nSamples,d,K);
      Sigma = zeros(d,d,nSamples,K);
      for k=1:K
        mu(:,:,k) = muDist{k}.samples';
        Sigma(:,:,:,k) = SigmaDist{k}.samples;
      end
      % Save lognormconst
      invS = zeros(d,d,nSamples,K);
      for s=1:nSamples
        for k=1:K
          invS(:,:,s,k) = inv(Sigma(:,:,s,k));
          logconst(s,k) = 1/2*logdet(2*pi*Sigma(:,:,s,k));
        end
      end

      % perm will contain the permutation that minimizes step two of the algorithm
      % The permutations are as indices, is perm(1,1) indicates how we permute label 1 for iteration 1
      perm = bsxfun(@times,1:K,ones(nSamples,K));
      ident = bsxfun(@times, 1:K, ones(nSamples,K));
      oldPerm = bsxfun(@times,-inf,ones(nSamples,K));
      fixedPoint = false;
      % For tracking purposes; value is how many times we have done the algorithm (itr and k already taken)
      klscore = 0;
      value = 1;
      fail = false;
      fprintf('Attempting to resolve label switching \n')
      while( ~fixedPoint )
        if(verbose), fprintf('(Run %d). Computing Q for iteration  ', value);end;
        Q = zeros(nObs,K); logpdata = zeros(nObs,nSamples,K);
        %oldPerm = perm;
        % Note that doing both qmodel and pmodel in the same loop is more efficient in terms of runtime
        % but we need to store pij for each iteration.  This can cause memory to run out if 
        % either nobs or iter is large.
        % Hence, we first compute Q and then pij.
        for itr = 1:nSamples
          logqRik = zeros(nObs,K);
          for k=1:K
            %XC = bsxfun(@minus, X, mu(itr,:,oldPerm(itr,k)));
            %logpdata(:,itr,k) = -logconst(itr,oldPerm(itr,k)) - 1/2*sum((XC*invS(:,:,itr,oldPerm(itr,k))).*XC,2);
            %logqRik(:,k) = log(mix(itr,oldPerm(itr,k))+eps)+ logpdata(:,itr,k);
            XC = bsxfun(@minus, X, mu(itr,:,k));
            logpdata(:,itr,k) = -logconst(itr,k) - 1/2*sum((XC*invS(:,:,itr,k)).*XC,2);
            logqRik(:,k) = log(mix(itr,k))+ logpdata(:,itr,k);
          end
          Q = Q + exp(normalizeLogspace(logqRik));
        end
        Q = Q / nSamples;
        if(verbose)
          fprintf('computed.  Optimizing over permutations.  ')
        end
        % Loss for each individual iteration
        loss = zeros(nSamples,1);
        for itr = 1:nSamples
          logpij = zeros(nObs,K);
          for k=1:K
            %logpij(:,k) = log(mix(itr,oldPerm(itr,k))+eps) + logpdata(:,itr,oldPerm(itr,k));
            logpij(:,k) = log(mix(itr,k)) + logpdata(:,itr,k);
          end
          logpij = normalizeLogspace(logpij);
          pij = exp(logpij);
          kl = zeros(K,K);
          for j=1:K
            for l=1:K
              diverge = pij(:,l).* log(pij(:,l) ./ Q(:,j));
              % We want 0*log(0/q) = 0 for q > 0 (definition of 0*log(0) for KL)
              diverge(isnan(diverge)) = 0;
              kl(j,l) = sum(diverge);
            end
          end
        % find the optimal permutation for this iteration, and then store in loss vector
        [perm(itr,:), loss(itr)] = assignmentoptimal(kl);
        if(any(perm(itr,:) == 0)), keyboard, end;
        end
        % KL loss is the sum of all the losses over all the iterations
        klscore(value) = mean(loss);

        if(value == 1)
          deltakl = 0;
        else
          deltakl = klscore(value) - klscore(value - 1);
        end
        if(verbose),fprintf('KL Loss = %g.  Delta = %g \n',klscore(value), deltakl); end;
        % Stopping criteria - what would be ideal is to have a vector of stopping criteria
        % and have the user select the stopping criteria
        % I'm thinking that we could pass this in as varargin, and then evaluate the chosen
        % criteria after each run
        %if( value > 2 && (all(all(perm == oldPerm)) || approxeq(klscore(value), klscore(value-1), 1/klscore(value), 1) || approxeq(klscore(value), klscore(value-2), 1/klscore(value), 1) ) )
        %if(value > 2 && (all(all(perm == ident)) || value >= maxitr || convergenceTest(klscore(value), klscore(value - 1))))
        if(value > 2 && (all(all(perm == oldPerm)) || value >= maxitr || convergenceTest(klscore(value), klscore(value - 1))))
          fixedPoint = true;
        end
        if(deltakl > 0)
          warning('Objective did not decrease.  Returning with last good permutation');
          permOut = oldPerm;
          fixedPoint = true;
          fail = true;
        end

        if(fail ~= true && ~fixedPoint)
          for k=1:K
            mu(itr,:,k) = mu(itr,:,perm(itr,k));
            logconst(itr,k) = logconst(itr,perm(itr,k));
            invS(:,:,itr,k) = invS(:,:,itr,perm(itr,k));
          end
        end
        value = value + 1;   
      end
      permOut = perm;
      latentout = zeros(nSamples,nObs);
      mixout = zeros(nSamples,K);
      muout = zeros(nSamples,d,K);
      Sigmaout = zeros(d,d,nSamples,K);
      for itr=1:nSamples
        latentout(itr,:) = permOut(itr,latent(itr,:));
        mixout(itr,:) = mix(itr,permOut(itr,:));
        for k=1:K
          muout(itr,:,k) = mu(itr,:,permOut(itr,k));
          Sigmaout(:,:,itr,k) = Sigma(:,:,itr,permOut(itr,k));
        end
      end
      latentoutDist = SampleDist(latentout');
      mixoutDist = SampleDist(mixout');
      muoutDist = cell(K,1); SigmaoutDist = cell(K,1);
      for k=1:K
        muoutDist{k} = SampleDist(muout(:,:,k)');
        SigmaoutDist{k} = SampleDist(Sigmaout(:,:,:,k)); 
      end 
      distsout = struct;%('muDist', muoutDist, 'SigmaDist', SigmaoutDist, 'mixDist', mixoutDist, 'latentDist', latentoutDist);
      distsout.muDist = muoutDist;
      distsout.SigmaDist = SigmaoutDist;
      distsout.mixoutDist = mixoutDist;
      distsout.latentDist = latentoutDist;
      fprintf('\n')
    end
