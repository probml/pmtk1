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