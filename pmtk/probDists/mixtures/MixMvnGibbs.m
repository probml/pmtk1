classdef MixMvnGibbs < ProbDist
 
  % Gibbs sampling for a mixture of multivariate normals
  
  properties
    burnin;
    nchains;
    nsamples; 
    % we do not store distributions or mixingDistrib.
    % Instead, we store posterior over their parameters, represented
    % as samples.
    priors; % priors.muSigmaPrior shared across k, priors.mixPrior
    samples; % samples.mu(:,k,s), samples.Sigma(:,:,k,s), samples.pi(k,s)
  end

  methods

    function model = MixMvnGibbs(varargin)
      % model = MixMvn(...)
      % Create a model with default priors for MAP estimation
      [nmixtures,  transformer, verbose, nsamples, nchains, burnin]...
        = processArgs(varargin,...
        'nmixtures'    ,[] ,...
        'transformer'  ,[], ...
        'verbose',      true, ...
        'nsamples',      100,...
        'nchains',       3,...
        'burnin',      1000);
      K = nmixtures;
      mixingDistrib = DiscreteDist('T', normalize(rand(K,1)), ...
        'prior', DirichletDist(normalize(ones(K,1))));
      dist = MvnDist([],[],'prior','niw');
      distributions = copy(dist,K,1);
      model = MixtureModel(K, distributions, mixingDistrib, transformer);
      model.verbose = verbose;
      model.nsamples = nsamples;
      model.nchains = nchains;
      model.burnin = burnin;
    end

    function model = fit(model, data)
      error('not yet implemented')
      model.samples = runMCMC;
    end
    
    function [ph, LL] = conditional(model,data)
      % ph(i,k) = (1/S) sum_s p(H=k | data(i,:),params(s)), a DiscreteDist
      % This is the posterior responsibility of component k for data i
      % LL(i) = log p(data(i,:) | params)  is the log normalization const
      S = model.nsamples;
      K = nmixtures(model);
      N = size(data,1);
      Rik = zeros(N,K);
      for s=1:S
        mixWeight = model.samples.pi(:,s);
        RikS = zeros(N,K); % responsibility given parameter s
        for k=1:K
          RikS(:,k) = mvnpdf(data, model.samples.mu(:,k,s), model.samples.Sigma(:,:,k,s));
          RikS(:,k) = Rik(:,k) * mixWeight(k);
        end
        RikS = normalize(Rik);
        Rik = Rik + RikS;
      end
      Rik = Rik/S;
      ph = DiscreteDist('T',Rik');
      LL = []; % not yet supported
    end

     function logp = logprob(model,data)
      % logp(i) = log sum_s p(data(i,:) | params(s))
      %  = log sum_s sum_k p(data(i,:), h=k | params(s)) 
      
      % just average over samples, similar to conditional above
      %logp = logsumexp(calcResponsibilities(model, data),2);
     end
    


      %%%%%%%% Cody code  below
    function [mcmc] = latentGibbsSample(model,data,varargin)
      [Nsamples, Nburnin, thin, Nchains, verbose] = process_options(varargin, ...
        'Nsamples'	, 1000, ...
        'Nburnin'		, 500, ...
        'thin'		, 1, ...
        'verbose'		, false );
      % Initialize the model
      [model, prior] = initializeGibbs(model,data);
      K = numel(model.distributions);
      [n,d] = size(data);

      % Create a structure that will store the required information
      obs = ceil((Nsamples - Nburnin) / thin);
      mcmc.latent = zeros(n,obs);
      mcmc.loglik = zeros(1,obs);
      mcmc.mix = zeros(K,obs);
      mcmc.param = cell(K,obs);
      keep = 1;
      % Get samples
      for itr=1:Nsamples
        % Contains pred.mu(k,i) = p(Z_k | data(i,:), params).  We sample an instance from this
        pred = predict(model,data);
        %mcmc.latent(:,itr) = colvec(sample(pred,1));
        latent = colvec(sample(pred,1));
        % Sample the parameters of the model given the assignment
        %[model, mcmc.param{itr}] = sampleParamGibbs(model,data,mcmc.latent(:,itr));
        param = cell(K,1);
        for k=1:K
          [model.distributions{k}, sampledparam] = sampleParamGibbs(model.distributions{k}, data(latent == k,:), prior{k});
          param{k} = sampledparam;
        end
        postMix = fit(Discrete_DirichletDist(model.mixingWeights.prior), 'data', latent);
        mix = sample(postMix.muDist, 1);

        %mcmc.loglik(itr) = logprobGibbs(model,data,mcmc.latent(:,itr));
        loglik = logprobGibbs(model,data,latent);
        if(itr > Nburnin && mod(itr, thin)==0)
          mcmc.latent(:,keep) = latent;
          mcmc.param(:,keep) = param;
          mcmc.mix(:,keep) = mix;
          mcmc.loglik(keep) = logprobGibbs(model,data,latent);
          keep = keep + 1;
        end
      end % of itr=1:Nsamples
    end



    function logp = logprobGibbs(model,data,latent)
      % logp(i) = log p(data(i,:) | params, latent(i) = k)
      K = numel(model.distributions);
      logp =  0;
      for k=1:K
        logp = logp + sum( logprob(model.distributions{k}, data(latent == k,:)) );
      end
    end

    function [permOut] = processLabelSwitch(model,mcmc,X)
      warning('MixtureDist:processLabelSwitch:notComplete', 'Warning, function not fully debugged yet.  May contain errors')
      fprintf('Waring.  Post-processing of latent variable to compensate for label-switching can take a great deal of time (iterative method involving a search over K! permutations for mixture models with K components).  Please be patient. \n')
      N = numel(mcmc.loglik);
      n = size(mcmc.latent,1);
      K = size(mcmc.mix,1);
      % perm contains the N permutations that we consider.
      % The permutations are as indices, is perm(1,1) indicates how we permute label 1 for iteration 1
      perm = bsxfun(@times,1:K,ones(N,K));
      Q = zeros(n,K);
      oldPerm = bsxfun(@times,-inf,ones(N,K));
      while( any(oldPerm ~= perm) )
        fprintf('Computing / Selecting Q matrix.\n')
        oldPerm = perm;
        % Recreate the models
        for itr = 1:N
          tmpmodel = model;
          SS.counts = mcmc.latent(:,itr);
          tmpmodel.mixingWeights = setParams(model.mixingWeights, mcmc.mix(:,itr) );
          for k=1:K
            tmpmodel.distributions{k} = setParams(model.distributions{k}, mcmc.param{perm(itr,k),itr});
          end
          Q = Q + exp(normalizeLogspace(calcResponsibilities(tmpmodel,X)));
        end
        Q = Q ./ N;

        allperm = perms(1:K);
        klloss = zeros(factorial(K),1);
        prob = zeros(n,K);
        permmodel = model;
        fprintf('Selecting permutation on label to minimize KL-loss \n')
        for itr=1:N
          fprintf('%d, ', itr)
          for j = 1:factorial(K)
            permmodel.mixingWeights = setParams(permmodel.mixingWeights, mcmc.mix(:,itr));
            %SS.counts = mcmc.latent;
            %permmodel.mixingWeights = fit(permmodel.mixingWeights, 'suffStat', SS);
            for k=1:K
              permmodel.distributions{k} = setParams(model.distributions{k}, mcmc.param{allperm(j,k),itr} );
            end
          end
          pij = normalizeLogspace(calcResponsibilities(permmodel,X));
          klloss(itr) = sum(sum( pij.*log(pij ./ Q) ));
          perm(itr,:) = allperm(argmin(klloss(itr)),:);
        end
      end
      permOut = perm;
      fprintf('\n')
    end



    function [model,prior] = initializeGibbs(model,X)
      % we initialize by partitioning the observations into the K mixture components at random
      % we return (initialized) priors for each model
      K = numel(model.distributions);
      [n,d] = size(X);
      group = Kfold(n ,K);
      prior = cell(K,1);
      for k=1:K
        model.distributions{k} = mkRndParams( model.distributions{k},d );
        switch class(model.distributions{k}.prior)
          case 'char'
            prior{k} = mkPrior(model.distributions{k},X);
          otherwise
            prior{k} = model.distributions{k}.prior;
        end
      end
    end


  end % methods

end

