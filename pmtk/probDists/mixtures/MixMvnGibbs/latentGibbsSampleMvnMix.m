
function [dists] = latentGibbsSampleMvnMix(model,X,varargin)
[Nsamples, Nburnin, thin, verbose] = process_options(varargin, ...
  'Nsamples'	, 1000, ...
  'Nburnin'		, 500, ...
  'thin'		, 1, ...
  'verbose'		, false );

% Initialize the model
K = numel(model.distributions);
[n,d] = size(X);
[model, prior, priorlik] = initializeGibbs(model,X);
post = cell(K,1);
if(verbose)
  fprintf('Gibbs Sampling initiated.  Starting to collect samples\n')
end

obs = ceil((Nsamples - Nburnin) / thin);
loglik = zeros(obs,1);
latentsamples = zeros(obs,n);
musamples = zeros(obs,d,K);
Sigmasamples = zeros(d,d,obs,K);
mixsamples = zeros(obs,K);

keep = 1;
% Get samples
for itr=1:Nsamples
  if(mod(itr,500) == 0 && verbose)
    fprintf('Collected %d samples \n', itr)
  end
  % sample the latent variables conditional on the parameters
  pred = predict(model,X);
  latent = colvec(sample(pred,1));
  % Sample the parameters of the model conditional on the parameters
  param = cell(K,1);
  for k=1:K
    joint = fit(priorlik{k},'data', X );
    post{k} = joint.muSigmaDist;
    switch class(prior{k})
      case 'MvnInvWishartDist'
        % From post, get the values that we need for the marginal of Sigma for this distribution, and sample
        postSigma = InvWishartDist(post{k}.dof + 1, post{k}.Sigma);
        model.distributions{k}.Sigma = sample(postSigma,1);
        % Now, do the same thing for mu
        postMu = MvnDist(post{k}.mu, model.distributions{k}.Sigma / post{k}.k);
      case 'MvnInvGammaDist'
        postSigma = InvGammaDist(post.a + 1, post.b);
        model.distributions{k}.Sigma = sample(postSigma,1);
        % Now, do the same thing for mu
        postMu = MvnDist(post.mu, obj.Sigma / post.Sigma);
    end % of switch class(prior)
    model.distributions{k}.mu = sample(postMu,1);
  end % of k=1:K
  % Conditional on the sampled assignments, fit and sample from the Dirichlet-Multinomial that defines the mixing weights
  postMix = fit(Discrete_DirichletDist(model.mixingWeights.prior), 'data', latent);
  model.mixingWeights.T = sample(postMix.muDist, 1);

  % Store the results if we are past burnin and if thinning permits
  if(itr > Nburnin && mod(itr, thin)==0)
    latentsamples(keep,:) = rowvec(latent);
    mixsamples(keep,:) = rowvec(model.mixingWeights.T);
    for k=1:K
      musamples(keep,:,k) = rowvec(model.distributions{k}.mu);
      Sigmasamples(keep,:,k) = rowvec(cholcov(model.distributions{k}.Sigma));
      if(Sigmasamples(keep,:,k) == 0)
        fprintf('invalid Sigma')
        keyboard
      end
    end
    loglik(keep) = logprobGibbs(model,X,latent);
    keep = keep + 1;
  end
end % of itr=1:Nsamples
latentDist = SampleDistDiscrete(latentsamples, 1:K);
mixDist = SampleDistDiscrete(mixsamples, 1:K);
muDist = SampleDist(musamples, 1:d);
% from the documentation, I'm not exactly sure how to storethe covariance matrix samples.
% Suggest storing the cholesky factor as a vector.  Can recover original matrix using
% reshape(w',4,4)'*reshape(w',4,4), where w is the sample cholesky factor
SigmaDist = SampleDist(Sigmasamples);
dists = struct('latentDist', latentDist, 'muDist', muDist, 'SigmaDist', SigmaDist, 'mixDist', mixDist);
end

