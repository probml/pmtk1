function dists = collapsedGibbsSampleMvnMix(distributions, mixingWeights, data, varargin)
  % Collapsed Gibbs sampling for a mixture of MVNs
  [Nsamples, Nburnin, thin, verbose] = process_options(varargin, ...
    'Nsamples'  , 1000, ...
    'Nburnin'   , 500, ...
    'thin'    , 1, ...
    'verbose'   , true );
  [nobs,d] = size(data); K = numel(distributions);
  outitr = ceil((Nsamples - Nburnin) / thin); keep = 1;
  latent = zeros(outitr, nobs);
  alpha = mixingWeights.prior.alpha;
  if(verbose), fprintf('Initializing Gibbs sampler... \n'); end;
  
  % latent is the column vector of intial assignments
  % prior{k} is the prior for distribution k
  % rest are sufficient statistics in the form of matrices (for more efficient computation)
  latent(1,:) = unidrnd(K,1,nobs);
  priorMuSigmaDist = cell(K,1);
  SSn = zeros(K,1); SSxbar = zeros(d,K); SSXX = zeros(d,d,K); SSXX2 = zeros(d,d,K);
  covtype = cell(1,K);

  % since each MVN could have a different covariance structure, we need to do this.
  for k=1:K
    switch class(distributions{k}.prior)
      case 'char'
      priorMuSigmaDist{k} = mkPrior(distributions{k},'-data', data);
      otherwise
      priorMuSigmaDist{k} = distributions{k}.prior;
    end
      [distributions{k}.mu, distributions{k}.Sigma] = sample(priorMuSigmaDist{k});
      SS{k} = MvnDist().mkSuffStat(data(latent(1,:)' == k,:));
      SSn(k) = SS{k}.n;
      SSxbar(:,k) = SS{k}.xbar;
      SSXX(:,:,k) = SS{k}.XX;
      SSXX2(:,:,k) = SS{k}.XX2;;
      priorlik{k} = MvnConjugate(distributions{k}, 'prior', priorMuSigmaDist{k});
      covtype{k} = distributions{k}.covtype;
  end

  curlatent = colvec(latent(1,:));


  % Collect samples
  if(verbose), fprintf('Samples collected: '); end;
  for itr=1:Nsamples
    if(verbose && mod(itr,100) == 0), fprintf('%d... ', itr); end;
    for obs=1:nobs
    prob = zeros(1,K);
      % For each iteration and observation, get the current assignment of the current observation
      % update the posterior parameters for its current cluster to reflect the absence of this observation
      kobs = curlatent(obs);
      xi = data(obs,:);
      [SSxbar, SSn, SSXX, SSXX2] = SSremoveXi(SSxbar, SSn, SSXX, SSXX2, kobs, xi);
      for k = 1:K
          if(any(~isfinite(SSxbar)) || any(~isfinite(SSXX(:,:,k))) || any(~isfinite(SSXX2(:,:,k)))),keyboard;end;
          [k0,v0,S0,m0] = updatePost(priorMuSigmaDist{k}, SSxbar, SSn, SSXX, SSXX2, k, xi);
          kn = k0 + 1;
          mn = (k0*rowvec(m0) + xi)/kn;
          % In previous version of this code, we worked with objects that were alway of type
          % MvtDist or StudentDist, updated the parameters of these objects, and then called logProb.
          % This was slow, since logProb did a lot more than we needed to actually do.
          % The new version of the code computes the necessary marginal probability directly
          % Credit to Matt Dunham for the suggestion
          switch covtype{k}
            case 'full'
              vn = v0 + 1;
              Sn = S0 + k0/(k0+1)*(xi'-colvec(m0))*(xi'-colvec(m0))';
              logZ = (d/2)*(log(k0) - log(kn) - log(pi)) + mvtGammaln(d,vn/2) - mvtGammaln(d,v0/2);
              dist = (v0/2)*logdet(S0) - (vn/2)*logdet(Sn);
              logmprob = dist + logZ;
            case 'spherical'
              an = v0 + d/2;
              bn = S0 + 1/2*sum(diag( k0/(k0+1)*(xi'-colvec(m0))*(xi'-colvec(m0))' ));
              logZ = (d/2)*(log(k0) - log(kn) - log(2*pi)) + gammaln(an) - gammaln(v0);
              dist = v0*logdet(b) - an*logdet(bn);
              logmprob = dist + logZ;
            case 'diagonal'
              an = v0 + 1/2;
              bn = diag( diag(S0) + 1/2*(k0/(k0+1)*(xi'-colvec(m0))*(xi'-colvec(m0))'));
              logZ = (d/2)*(log(k0) - log(kn) - log(2*pi)) + sum(gammaln(an) - gammaln(v0))
              dist = v0.*gammaln(b0) - an.*gammaln(bn);
              logmprob = dist + logZ;
          end
        lognij = log(SSn(k) + alpha(k));
        if(isnan(lognij) || isnan(logmprob)),keyboard;end;
        % Now find the probability of this observation being in each cluster
        prob(k) =  lognij + logmprob;
      end % for clust = 1:K
      % normalize, sample, and then update the sufficient statistics to reflect the new assignment
      prob = colvec(exp(normalizeLogspace(prob)));
      curlatent(obs) = sample(prob,1);
      [SSxbar, SSn, SSXX, SSXX2] = SSaddXi(SSxbar, SSn, SSXX, SSXX2, curlatent(obs), data(obs,:));
    end % obs=1:n
    % Store the results if past burnin and if thinning permits
    if(itr > Nburnin && mod(itr, thin)==0)
      latent(keep,:) = rowvec(curlatent);
      keep = keep + 1;
    end
  end
  if(verbose)
    fprintf('\n')
  end

  % With Gibbs sampling complete, create the mcmc struct needed for returning
  latentDist = SampleDistDiscrete(latent, 1:K);

  mixDist = mixingWeights;

  musamples = zeros(outitr, d, K);
  Sigmasamples = zeros(outitr, d*d, K);

  mcmc.loglik = zeros(1,outitr);
  mix = zeros(outitr,K);
  tmpdistrib = distributions;
  for itr=1:outitr
    mixFit = fit( mixDist, '-data', latent(itr,:)' );
    mix(itr,:) = rowvec(mixFit.T);
    for k=1:K
      [mu, Sigma, domain] = convertToMvn( fit(distributions{k}, '-data', data(latent(itr,:)' == k,:) ) );
      musamples(itr,:,k) = mu;
      Sigmasamples(itr,:,k) = rowvec(cholcov(Sigma));
    end
  end
  muDist = SampleDist(musamples, 1:d);
  SigmaDist = SampleDist(Sigmasamples);
  mixDist = SampleDistDiscrete(mix, 1:K);
  dists = struct('muDist', muDist, 'SigmaDist', SigmaDist, 'mixDist', mixDist, 'latentDist', latentDist);
end

function [latent, prior, SSxbar, SSn, SSXX, SSXX2] = initializeCollapsedGibbs(model,X)
      % latent is the column vector of intial assignments
      % prior{k} is the prior for distribution k
      % rest are sufficient statistics in the form of matrices (for more efficient computation)
      [n,d] = size(X);
      K = numel(model.distributions);
      latent = unidrnd(K,n,1);
      %latent = kmeans(X,K);
      prior = cell(K,1);

      SSn = zeros(K,1);
      SSxbar = zeros(d,K);
      SSXX = zeros(d,d,K);
      SSXX2 = zeros(d,d,K);

      % since each MVN could have a different covariance structure, we need to do this.
      for k=1:K
        switch class(model.distributions{k}.prior)
          case 'char'
            prior{k} = mkPrior(model.distributions{k},'data', X);
          otherwise
            prior{k} = model.distributions{k}.prior;
        end
            [model.distributions{k}.mu, model.distributions{k}.Sigma] = sample(prior{k});
            SS{k} = MvnDist().mkSuffStat(X(latent == k,:));
            SSn(k) = SS{k}.n;
            SSxbar(:,k) = SS{k}.xbar;
            SSXX(:,:,k) = SS{k}.XX;
            SSXX2(:,:,k) = SS{k}.XX2;;
            priorlik{k} = MvnConjugate(model.distributions{k}, 'prior', prior{k});
      end
    end % initializeCollapsedGibbs

    
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
    
    function [kn,vn,Sn,mn] = updatePost(priorMuSigmaDist, SSxbar, SSn, SSXX, SSXX2, k, xi)
      if(SSn(k) == 0)
          switch class(priorMuSigmaDist)
            case 'MvnInvWishartDist'
              kn = priorMuSigmaDist.k; mn = priorMuSigmaDist.mu;
              Sn = priorMuSigmaDist.Sigma; vn = priorMuSigmaDist.dof;
            case 'MvnInvGammaDist'
              kn = priorMuSigmaDist.Sigma; mn = priorMuSigmaDist.mu;
              Sn = priorMuSigmaDist.b; vn = priorMuSigmaDist.a;
          end
      return;
      end
      % returns parameters reflecting the loss of xi from cluster k
        n = SSn(k); d = length(xi);
        xbar =  SSxbar(:,k);
          switch class(priorMuSigmaDist)
            case 'MvnInvWishartDist'
              k0 = priorMuSigmaDist.k; m0 = priorMuSigmaDist.mu;
              S0 = priorMuSigmaDist.Sigma; v0 = priorMuSigmaDist.dof;
              kn = k0 + n;
              vn = v0 + n;
              Sn = S0 + n*SSXX(:,:,k) + (k0*n)/(k0+n)*(xbar-colvec(m0))*(xbar-colvec(m0))';
              mn = (k0*colvec(m0) + n*xbar)/kn;
            case 'MvnInvGammaDist'
              k0 = priorMuSigmaDist.Sigma; m0 = priorMuSigmaDist.mu;
              S0 = priorMuSigmaDist.b; v0 = priorMuSigmaDist.a;
              mn = (k0*colvec(m0) + n*xbar)/kn;
              switch lower(covtype)
                case 'spherical'
                  vn = v0 + n*d/2;
                  Sn = S0 + 1/2*diag(sum(diag( n*SSXX(:,:,k) + (k0*n)/(k0+n)*(xbar-colvec(m0))*(xbar-colvec(m0))' )));
                case 'diagonal'
                  vn = v0 + n/2;
                  Sn = diag( diag(S0) + 1/2*(n*SSXX(:,:,k) + (k0*n)/(k0+n)*(xbar-colvec(m0))*(xbar-colvec(m0))'));
              end
          end
    end
