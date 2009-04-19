classdef MixtureDistOld < ParamJointDist
  % A mixture model
  % Mixtures of any probability distributions can be created with this class.
  % Subclasses simply allow for more a more convienient interface by automatically
  % instantiating the mixture components. To fit via EM, however, each mixture
  % component must have a mkSuffStat method and a 'suffStat' option to fit().

  properties
    distributions;      % a cell array storing the distributions
    mixingWeights;      % DiscreteDist or Discrete_DirichletDist
    verbose = false;
    transformer;        % data preprocessor
    nrestarts = 5;      % number of random restarts
  end

  methods

    function model = MixtureDistOld(varargin)
      % Construct a new mixture of distributions
      if nargin == 0, return; end;
      [nmixtures,distributions,mixingWeights,model.transformer]...
        = process_options(varargin,...
        'nmixtures'    ,[] ,...
        'distributions',[] ,...
        'mixingWeights',[] ,...
        'transformer'  ,[]);
      
      if ~isempty(nmixtures) && numel(distributions) == 1
        distributions = copy(distributions,nmixtures,1);
      end
      if isempty(nmixtures)
        nmixtures = numel(distributions);
      end
      model.distributions = distributions;
      if(~isempty(nmixtures) && isempty(mixingWeights))
        mixingWeights = DiscreteDist('T',normalize(ones(nmixtures,1)),'support',1:nmixtures);
      elseif(~isempty(model.distributions))
        mixingWeights = DiscreteDist('T',normalize(ones(numel(model.distributions,1))),'support',1:nmixtures);
      end
      model.mixingWeights = mixingWeights;
    end

    function p = isDiscrete(CPD) %#ok
      p = false;
    end

    function q = nstates(CPD)  %#ok
      q = length(CPD.mixingWeights);
    end


    function model = fit(model,varargin)
      % Fit via EM
      [data,opttol,maxiter,nrestarts,SS,prior,init] = process_options(varargin,...
        'data'      ,[]    ,...
        'opttol'    ,1e-3  ,...
        'maxiter'   ,20    ,...
        'nrestarts' ,model.nrestarts ,...
        'suffStat', []     ,...
        'prior'     ,'none',...
        'init'      , true );
      nmixtures = numel(model.distributions);
      if(not(isempty(SS))),fitSS();return;end
      if(~isempty(model.transformer))
        [data,model.transformer] = train(model.transformer,data);
      end
      if(init),model = initializeEM(model,data);end
      bestDists = model.distributions;
      bestMix   = model.mixingWeights;
      bestLL    = sum(logprob(model,data));
      bestRR    = 1;
      for r = 1:nrestarts,
        emUpdate();
      end
      model.distributions = bestDists;
      model.mixingWeights = bestMix;
      if(model.verbose),displayProgress(model,data,bestLL,bestRR);end
      %% Sub functions to fit
      function fitSS()
        % Fit via sufficient statistics
        for k=1:nmixtures
          model.distributions{k} = fit(model.distributions{k},'suffStat',SS.ess{k},'prior',prior);
        end
        mixSS.counts = colvec(normalize(sum(SS.weights,1)));
        model.mixingWeights = fit(model.mixingWeights,'suffStat',mixSS);
      end

      function emUpdate()
        % Perform EM
        if(r>1 && init),model = initializeEM(model,data);end
        converged = false; iter = 0; currentLL =sum(logprob(model,data));
        while(not(converged))
          if(model.verbose),displayProgress(model,data,currentLL,r);end
          prevLL = currentLL;
          logRik = calcResponsibilities(model,data);  % responsibility of each cluster for each data point
          Rik = exp(bsxfun(@minus,logRik,logsumexp(logRik,2))); % normalize
          for k=1:nmixtures
            ess = model.distributions{k}.mkSuffStat(data,Rik(:,k));
            model.distributions{k} = fit(model.distributions{k},'suffStat',ess,'prior',prior);
          end
          mixSS.counts = colvec(normalize(sum(Rik,1)));
          model.mixingWeights = fit(model.mixingWeights,'suffStat',mixSS);
          iter = iter + 1;
          currentLL = sum(logprob(model,data));
          converged = iter >=maxiter || (abs(currentLL - prevLL) / (abs(currentLL) + abs(prevLL) + eps)/2) < opttol;
        end
        if(currentLL > bestLL)
          bestDists = model.distributions;
          bestMix   = model.mixingWeights;
          bestLL    = sum(logprob(model,data));
          bestRR    = r;
        end
        %%
      end % end of emUpdate() subfunction
    end % end of fit method

    function [mcmc] = latentGibbsSample(model,data,varargin)
      [Nsamples, Nburnin, thin, verbose] = process_options(varargin, ...
        'Nsamples'	, 1000, ...
        'Nburnin'		, 500, ...
        'thin'		, 1, ...
        'verbose'		, false );
      % Initialize the model
      [model, prior] = initializeGibbs(model,data);
      if(verbose)
        fprintf('Gibbs Sampling initiated.  Starting to collect samples\n')
      end
      K = numel(model.distributions);
      [n,d] = size(data);

      % Create a structure that will store the required information
      % Latent variables for each iteration are stored as column vectors, 
      % as are mixing weights and a struct containing the parameters for each mixing distribution
      obs = ceil((Nsamples - Nburnin) / thin);
      mcmc.latent = zeros(n,obs);
      mcmc.loglik = zeros(1,obs);
      mcmc.mix = zeros(K,obs);
      mcmc.param = cell(K,obs);
      keep = 1;
      % Get samples
      for itr=1:Nsamples
        if(mod(itr,500) == 0 && verbose)
          fprintf('Collected %d samples \n', itr)
        end

        % sample the latent variables conditional on the parameters
        pred = predict(model,data);
        latent = colvec(sample(pred,1));
        % Sample the parameters of the model conditional on the parameters
        param = cell(K,1);
        for k=1:K
          % NOTE: sampleParamGibbs is a function that *must* be implemented in each distributions class file
          % this function must take in data and the prior for the parameters
          % and then output the model with the sampled parameters, along with a struct containing the sampled parameters
          [model.distributions{k}, sampledparam] = sampleParamGibbs(model.distributions{k}, data(latent == k,:), prior{k});
          param{k} = sampledparam;
        end
        % Conditional on the sampled assignments, fit and sample from the Dirichlet-Multinomial that defines the mixing weights
        postMix = fit(Discrete_DirichletDist(model.mixingWeights.prior), 'data', latent);
        mix = sample(postMix.muDist, 1);

        % Store the results if we are past burnin and if thinning permits
        if(itr > Nburnin && mod(itr, thin)==0)
          mcmc.latent(:,keep) = latent;
          mcmc.param(:,keep) = param;
          mcmc.mix(:,keep) = mix;
          mcmc.loglik(keep) = logprobGibbs(model,data,latent);
          keep = keep + 1;
        end
      end % of itr=1:Nsamples
    end

    function pred = predict(model,data)
      % pred.mu(k,i) = p(Z_k | data(i,:),params)
      logRik = calcResponsibilities(model,data);
      %Rik = exp(bsxfun(@minus,logRik,logsumexp(logRik,2)));
      Rik = exp(normalizeLogspace(logRik));
      pred = DiscreteDist('T',Rik');
    end

    function logp = logprob(model,data)
      % logp(i) = log p(data(i,:) | params)
      logp = logsumexp(calcResponsibilities(model,data),2);
    end

    function logp = logprobGibbs(model,data,latent)
      % logp(i) = log p(data(i,:) | params, latent(i) = k)
      K = numel(model.distributions);
      logp =  0;
      for k=1:K
        logp = logp + sum( logprob(model.distributions{k}, data(latent == k,:)) );
      end
    end

    function Tfac = convertToTabularFactor(model, child, ctsParents, dParents, visible, data, nstates,fullDomain)
      %function Tfac = convertToTabularFactor(model, domain,visVars,visVals)
      % domain = indices of each parent, followed by index of child
      % all of the children must be observed
      assert(isempty(ctsParents))
      assert(length(dParents)==1)
      map = @(x)canonizeLabels(x,fullDomain);
      if visible(map(child))
        T = exp(calcResponsibilities(model,data(map(child))));
        Tfac = TabularFactor(T,dParents);
      else
        % barren leaf removal
        Tfac = TabularFactor(ones(1,nstates(map(dParents))), dParents);
      end
    end

    %{
function Tfac = convertToTabularFactor(model, child, ctsParents, dParents, visible, data, nstates);
%function Tfac = convertToTabularFactor(model, domain,visVars,visVals)
% domain = indices of each parent, followed by index of child
% all of the children must be observed
assert(isempty(ctsParents))
assert(length(dParents)==1)
assert(visible(child))
visVals = data(child);
if(isempty(visVars))
Tfac = TabularFactor(1,domain); return; % return an empty TabularFactor
end
pdom = domain(1); cdom = domain(2:end);
if ~isequal(cdom,visVars)
% If we have a mixture of factored bernoullis
% the factor would be all discrete, but we don't handle this
% case.
error('Not all of the children of this CPD were observed.');
end
T = exp(calcResponsibilities(model,visVals));
Tfac = TabularFactor(T,pdom); % only a factor of the parent now
end
    %}


    function model = mkRndParams(model, d,K)
      for i=1:K
        model.distributions{i} = mkRndParams(model.distributions{i},d);
      end
      model.mixingWeights = DiscreteDist('T',normalize(rand(K,1)));
    end

    function model = condition(model, visVars, visValues)
      % pass condition requests through to mixture components
      if nargin < 2
        visVars = []; visValues = [];
      end
      model.conditioned = true;
      model.visVars = visVars;
      model.visVals = visValues;
      for i=1:numel(model.distributions)
        model.distributions{i} = condition(model.distributions{i},visVars,visValues);
      end
    end

    function postQuery = marginal(model, queryVars)
      % keep only the queryVars mixture components - barren node removal
      if(numel(queryVars == 1))
        postQuery = model.distributions{queryVars};
      else
        model.distributions = model.distributions{queryVars};
        model.mixingWeights = marginal(model.mixingWeights,queryVars);
        postQuery = model;
      end
    end

    function S = sample(model,nsamples)
      if nargin < 2, nsamples = 1; end
      Z = sampleDiscrete(mean(model.mixingWeights)', nsamples, 1);
      d = ndimensions(model);
      S = zeros(nsamples, d);
      for i=1:nsamples
        S(i,:) = rowvec(sample(model.distributions{Z(i)}));
      end
    end

    function d = ndimensions(model)
      if(numel(model.distributions) > 0)
        d = ndimensions(model.distributions{1});
      else
        d = 0;
      end
    end

    function d = ndistrib(model)
      d = max(1,numel(model.distributions));
    end


    function SS = mkSuffStat(model,data,weights)
      % Compute weighted, (expected) sufficient statistics. In the case of
      % an HMM, the weights correspond to gamma = normalize(alpha.*beta,1)
      % We calculate gamma2 by combining alpha and beta messages with the
      % responsibilities - see equation 13.109 in pml24nov08
      if(nargin < 2)
        weights = ones(size(data,1));
      end
      if(~isempty(model.transformer))
        [data,model.transformer] = train(model.transformer,data);
      end
      logRik = calcResponsibilities(model,data);
      logGamma2 = bsxfun(@plus,logRik,log(weights+eps));           % combine alpha,beta,local evidence messages
      %logGamma2 = bsxfun(@minus,logGamma2,logsumexp(logGamma2,2)); % normalize while avoiding numerical underflow
      logGamma2 = normalizeLogspace(logGamma2);
      gamma2 = exp(logGamma2);
      nmixtures = numel(model.distributions);
      ess = cell(nmixtures,1);
      for k=1:nmixtures
        ess{k} = model.distributions{k}.mkSuffStat(data,gamma2(:,k));
      end
      SS.ess = ess;
      SS.weights = gamma2;
    end

    function [mcmc,permOut] = processLabelSwitch(model,mcmc,X,varargin)
      % Implements the KL - algorithm for label switching from 
      %@article{ stephens2000dls,
      %	title = "{Dealing with label switching in mixture models}",
      %	author = "M. Stephens",
      %	journal = "Journal of the Royal Statistical Society. Series B, Statistical Methodology",
      %	pages = "795--809",
      %	year = "2000",
      %	publisher = "Blackwell Publishers"
      %}
      [verbose, stopCriteria] = process_options(varargin, 'verbose', false, 'stopCriteria', 1);
      N = numel(mcmc.loglik);
      n = size(mcmc.latent,1);
      K = size(mcmc.mix,1);

      % perm will contain the permutation that minimizes step two of the algorithm
      % oldPerm is the permutation that 
      % The permutations are as indices, is perm(1,1) indicates how we permute label 1 for iteration 1
      perm = bsxfun(@times,1:K,ones(N,K));
      oldPerm = bsxfun(@times,-inf,ones(N,K));
      fixedPoint = false;
      % For tracking purposes; value is how many times we have done the algorithm (itr and k already taken)
      klscore = 0;
      value = 1;
      while( ~fixedPoint )
        if(verbose)
          fprintf('Computing Q for iteration  ')
        end
        Q = zeros(n,K);
        oldPerm = perm;
        % Note that doing both qmodel and pmodel in the same loop is more efficient in terms of runtime
        % but we need to store pij for each iteration.  This can cause memory to run out if 
        % either nobs or iter is large.
        % Hence, we first compute Q and then pij.
        for itr = 1:N
          if(mod(itr,500) == 0),fprintf('%d, ', itr); end;
          logqRik = zeros(n,K);
          for k=1:K
            logqRik(:,k) = log(mcmc.mix(oldPerm(itr,k),itr)+eps)+ logprobParam(model.distributions{k}, X, mcmc.param{oldPerm(itr,k),itr});
          end
          Q = Q + exp(normalizeLogspace(logqRik));
        end
        Q = Q / N;
        if(verbose)
          fprintf('computed.  \n Optimizing over permutations.  ')
        end
        % Loss for each individual iteration
        loss = zeros(N,1);
        for itr = 1:N
          logpij = zeros(n,K);
          for k=1:K
            logpij(:,k) = log(mcmc.mix(oldPerm(itr,k),itr)+eps)+ logprobParam(model.distributions{k}, X, mcmc.param{oldPerm(itr,k),itr});
          end
          logpij = normalizeLogspace(logpij);
          pij = exp(logpij);
          kl = zeros(K,K);
          for j=1:K
            for l=1:K
              diverge = pij(:,l).* log(pij(:,l) ./ Q(:,j));
              % We want 0*log(0/q) = 0 for q > 0 (definition of 0*log(0) for KL
              diverge(isnan(diverge)) = 0;
              kl(j,l) = sum(diverge);
            end
          end
        % find the optimal permutation for this iteration, and then store in loss vector
        [perm(itr,:), loss(itr)] = assignmentoptimal(kl);
        end
        % KL loss is the sum of all the losses over all the iterations
        klscore(value) = sum(loss);

        % Stopping criteria - what would be ideal is to have a vector of stopping criteria
        % and have the user select the stopping criteria
        % I'm thinking that we could pass this in as varargin, and then evaluate the chosen
        % criteria after each run
        if( value > 2 && (all(all(perm == oldPerm)) || approxeq(klscore(value), klscore(value-1), 1e-2, 1) || approxeq(klscore(value), klscore(value-2), 1e-2, 1) ) )
          fixedPoint = true;
        end
      value = value + 1;
      if(verbose)
        fprintf('KL Loss = %d \n.',sum(loss))
      end   
      end
      permOut = perm;
      for itr=1:N
        mcmc.param(:,itr) = mcmc.param(permOut(itr,:),itr);
        mcmc.mix(:,itr) = mcmc.mix(permOut(itr,:),itr);
        mcmc.latent(:,itr) = permOut(itr,mcmc.latent(:,itr))';
      end

    end

  end


  methods(Access = 'protected')

    function model = initializeEM(model,X)
      % override in subclass if necessary
      for k=1:numel(model.distributions)
        model.distributions{k} = mkRndParams(model.distributions{k},X);
      end
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

    function displayProgress(model,data,loglik,rr)
      % override in subclass to customize displayed info
      if(model.verbose)
        fprintf('RR: %d, negloglik: %g\n',rr,-loglik);
      end
    end

    function logRik = calcResponsibilities(model,data)
      % returns unnormalized log responsibilities
      % logRik(i,k) = log(p(data(n,:),Z_k | params))
      % Used by predict(), logprob(), mkSuffStat()
      if(~isempty(model.transformer))
        data = test(model.transformer,data);
      end
      n = size(data,1); nmixtures = numel(model.distributions);
      logRik = zeros(n,nmixtures);
      for k=1:nmixtures
        logRik(:,k) = log(sub(mean(model.mixingWeights),k)+eps)+sum(logprob(model.distributions{k},data),2);
        % Calling logprob on vectorized distributions, (representing a
        % product, e.g. product of Bernoulli's) returns a matrix. We
        % therefore sum along the 2nd dimension in the 2nd term. This
        % has no effect for other distributions as logprob returns a
        % column vector.
      end
    end


  end





end

