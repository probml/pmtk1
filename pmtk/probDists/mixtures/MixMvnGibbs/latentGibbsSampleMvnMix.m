function [muS, sigmaS, mixS, latentS] = latentGibbsSampleMvnMix(distributions, mixingWeights, data, varargin)
  [Nsamples, Nburnin, thin, verbose] = processArgs(varargin, ...
    '-Nsamples'  , 1000, ...
    '-Nburnin'   , 500, ...
    '-thin'    , 1, ...
    '-verbose'   , false );
  
  % Initialize the model
  K = numel(distributions);
  [nobs,d] = size(data);
  prior = cell(K,1);
  priorlik = cell(K,1);
  for k=1:K
    if(isempty(distributions{k}.prior))
      distributions{k}.prior = mkPrior(distributions{k}, '-data', data, '-covtype', 'full', '-prior', 'niw');
    end
    switch class(distributions{k}.prior)
      case 'char'
        prior{k} = mkPrior(distributions{k},'-data', data);
      otherwise
        prior{k} = distributions{k}.prior;
    end
      [distributions{k}.mu, distributions{k}.Sigma] = sample(prior{k});
      priorlik{k} = MvnConjugate(distributions{k}, 'prior', prior{k});
  end
  
  mixPrior = mixingWeights.prior;
  post = cell(K,1);
  if(verbose)
    fprintf('Gibbs Sampling initiated.  Starting to collect samples\n')
  end
  
  outitr = ceil((Nsamples - Nburnin) / thin);
  latentsamples = zeros(outitr,nobs);
  musamples = zeros(outitr,d,K);
  Sigmasamples = zeros(outitr,d*d,K);
  mixsamples = zeros(outitr,K);

  keep = 1;
  logRik = zeros(nobs,K);
  % Get samples
  for itr=1:Nsamples
    if(mod(itr,500) == 0 && verbose)
      fprintf('Collected %d samples \n', itr)
    end
    % sample the latent variables conditional on the parameters
    for k=1:K
      logRik(:,k) = log(sub(mean(mixingWeights),k)+eps)+sum(logprob(distributions{k},data),2);
    end
    predictive = exp(normalizeLogspace(logRik));
    pred = DiscreteDist('-T',predictive');

    latent = colvec(sample(pred,1));
    for k=1:K
      joint = fit(priorlik{k}, 'data', data(latent == k,:) );
      post{k} = joint.muSigmaDist;
      switch class(prior{k})
        case 'MvnInvWishartDist'
          % From post, get the values that we need for the marginal of Sigma for this distribution, and sample
          sampledSigma = iwishrnd(post{k}.Sigma, post{k}.dof + 1);
          % Now, do the same thing for mu
          distributions{k}.mu = mvnrnd(rowvec(post{k}.mu), sampledSigma / post{k}.k);
        case 'MvnInvGammaDist'
          switch lower(covtype{k})
            case 'spherical'
              v = (post{k}.a + d/2)*2;
              s2 = 2*(post{k}.b)/v;
              sampledSigma = invchi2rnd(v,s2,1,1)*eye(d);
            case 'diagonal'
              v = (post{k}.a + 1/2)*2;
              s2 = 2*(post{k}.b)/v;
              for j=1:d
                sampledSigma(j) = invchi2rnd(v,s2(j),1,1);
              end
              sampledSigma = diag(sampledSigma);
          end
          distributions{k}.mu = mvnrnd(rowvec(post{k}.mu), sampledSigma / post{k}.Sigma);
      end % of switch class(prior)
      distributions{k}.Sigma = sampledSigma;
    end % of k=1:K
    % Conditional on the sampled assignments, fit and sample from the Dirichlet-Multinomial that defines the mixing weights
    postMix = fit(Discrete_DirichletDist(mixPrior), 'data', latent);
    mixingWeights.T = sample(postMix.muDist, 1);

    % Store the results if we are past burnin and if thinning permits
    if(itr > Nburnin && mod(itr, thin)==0)
      latentsamples(keep,:) = rowvec(latent);
      mixsamples(keep,:) = rowvec(mixingWeights.T);
      for k=1:K
        musamples(keep,:,k) = rowvec(distributions{k}.mu);
        Sigmasamples(keep,:,k) = rowvec(distributions{k}.Sigma);
      end
      keep = keep + 1;
    end
  end % of itr=1:Nsamples
  latentS = SampleBasedDist(latentsamples, 1:K);
  mixS = SampleBasedDist(mixsamples, 1:K);
  muS = cell(K,1); SigmaS = cell(K,1);
  for k=1:K
    muS{k} = SampleBasedDist(musamples(:,:,k), 1:d);
    SigmaS{k} = SampleBasedDist(Sigmasamples(:,:,k));      
  end
%  dists = struct;%;('muDist', muDist, 'SigmaDist', SigmaDist, 'mixDist', mixDist, 'latentDist', latentDist);
%  dists.muDist = muDist;
%  dists.SigmaDist = SigmaDist;
%  dists.mixDist = mixDist;
%  dists.latentDist = latentDist;
end

