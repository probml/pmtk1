classdef MvnMixDist < MixtureDist
  % Mixture of Multivariate Normal Distributions


  methods

    function model = MvnMixDist(varargin)
      [nmixtures,mixingWeights,distributions,model.transformer,model.verbose,model.nrestarts] = process_options(varargin,...
        'nmixtures',[],'mixingWeights',[],'distributions',[],'transformer',[],'verbose',true,'nrestarts',model.nrestarts);
      if(~isempty(distributions)),nmixtures = numel(distributions);end

      if(isempty(mixingWeights) && ~isempty(nmixtures))
        mixingWeights = DiscreteDist('T',normalize(ones(nmixtures,1)));
      end
      model.mixingWeights = mixingWeights;
      if(isempty(distributions)&&~isempty(model.mixingWeights))
        distributions = copy(MvnDist(),nstates(model.mixingWeights),1);
      end
      model.distributions = distributions;

    end

    function model = setParamsAlt(model, mixingWeights, distributions)
      model.mixingWeights = colvec(mixingWeights);
      model.distributions = distributions;
    end


    function model = mkRndParams(model, d,K)
      model.distributions = copy(MvnDist(),K,1);
      model = mkRndParams@MixtureDist(model,d,K);
    end

    function mu = mean(m)
      mu = zeros(ndimensions(m),ndistrib(m));
      for k=1:ndistrib(m)
        mu(:,k) = colvec(m.distributions{k}.mu);
      end

      M = bsxfun(@times,  mu, rowvec(mean(m.mixingWeights)));
      mu = sum(M, 2);
    end

    function C = cov(m)
      mu = mean(m);
      C = mu*mu';
      for k=1:numel(m.distributions)
        mu = m.distributions{k}.mu;
        C = C + sub(mean(m.mixingWeights),k)*(m.distributions{k}.Sigma + mu*mu');
      end
    end


    function xrange = plotRange(obj, sf)
      if nargin < 2, sf = 3; end
      %if ndimensions(obj) ~= 2, error('can only plot in 2d'); end
      mu = mean(obj); C = cov(obj);
      s1 = sqrt(C(1,1));
      x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
      if ndimensions(obj)==2
        s2 = sqrt(C(2,2));
        x2min = mu(2)-sf*s2; x2max = mu(2)+sf*s2;
        xrange = [x1min x1max x2min x2max];
      else
        xrange = [x1min x1max];
      end
    end

    function [muDist, SigmaDist, latentDist, mixDist] = latentGibbsSample(model,X,varargin)
      [Nsamples, Nburnin, thin, verbose] = process_options(varargin, ...
        'Nsamples'	, 1000, ...
        'Nburnin'		, 500, ...
        'thin'		, 1, ...
        'verbose'		, false );

      % Initialize the model
      K = numel(model.distributions);
      [n,d] = size(X);
      [model, prior, priorlik] = initializeGibbs(model,X);
      mixPrior = model.mixingWeights.prior;
      post = cell(K,1);
      if(verbose)
        fprintf('Gibbs Sampling initiated.  Starting to collect samples\n')
      end
      obs = ceil((Nsamples - Nburnin) / thin);
      loglik = zeros(obs,1);
      latentsamples = zeros(obs,n);
      musamples = zeros(obs,d,K);
      Sigmasamples = zeros(obs,d*d,K);
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
        for k=1:K
          joint = fit(priorlik{k}, 'data', X(latent == k,:) );
          post{k} = joint.muSigmaDist;
          switch class(prior{k})
            case 'MvnInvWishartDist'
              % From post, get the values that we need for the marginal of Sigma for this distribution, and sample
              postSigma = InvWishartDist(post{k}.dof + 1, post{k}.Sigma);
              sampledSigma = sample(postSigma,1);
              % Now, do the same thing for mu
              postMu = MvnDist(post{k}.mu, sampledSigma / post{k}.k);
            case 'MvnInvGammaDist'
              postSigma = InvGammaDist(post{k}.a + 1, post{k}.b);
              sampledSigma = sample(postSigma,1);
              % Now, do the same thing for mu
              % EXTREMELY inefficient based on profiling
              %postMu = MvnDist(post{k}.mu, sampledSigma / post{k}.Sigma);
          end % of switch class(prior)
          model.distributions{k}.Sigma = sampledSigma;
          model.distributions{k}.mu = mvnrnd(rowvec(post{k}.mu), sampledSigma / post{k}.Sigma);
          %model.distributions{k}.mu = sample(postMu,1);
        end % of k=1:K
%        keyboard
        % Conditional on the sampled assignments, fit and sample from the Dirichlet-Multinomial that defines the mixing weights
        postMix = fit(Discrete_DirichletDist(mixPrior), 'data', latent);
        model.mixingWeights.T = sample(postMix.muDist, 1);

        % Store the results if we are past burnin and if thinning permits
        if(itr > Nburnin && mod(itr, thin)==0)
          latentsamples(keep,:) = rowvec(latent);
          mixsamples(keep,:) = rowvec(model.mixingWeights.T);
          for k=1:K
            musamples(keep,:,k) = rowvec(model.distributions{k}.mu);
            Sigmasamples(keep,:,k) = rowvec(cholcov(model.distributions{k}.Sigma));
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
    % reshape(w',d,d)'*reshape(w',d,d), where w is the sample cholesky factor
    SigmaDist = SampleDist(Sigmasamples);      
    end

    function [muDist, SigmaDist, latentDist, mixDist] = collapsedGibbs(model,data,varargin)
      % Collapsed Gibbs sampling for a mixture of MVNs
      [Nsamples, Nburnin, thin, verbose] = process_options(varargin, ...
        'Nsamples'  , 1000, ...
        'Nburnin'   , 500, ...
        'thin'    , 1, ...
        'verbose'   , true );
      [nobs,d] = size(data);
      K = numel(model.distributions);
      outitr = ceil((Nsamples - Nburnin) / thin);
      latent = zeros(outitr, nobs);
      keep = 1;

      if(verbose)
        fprintf('Initializing Gibbs sampler... \n')
      end
      [latent(1,:), priorMuSigmaDist, priorlik, SSxbar, SSn, SSXX, SSXX2] = initializeCollapsedGibbs(model,data);
      curlatent = colvec(latent(1,:));
      % marginalDist stores objects for each observation, representing the marginal probabiltiy of each observation
      % being part of each cluster
      marginalDist = cell(nobs,K);
      % covtype will store the covariance structure for each cluster, since these need not be identical a priori
      covtype = cell(1,K);

      for c=1:K
        covtype{c} = model.distributions{c}.covtype;
        % Just create the marginalDist objects needed later on
        % Just create the marginalDist objects needed later on
        switch class(priorMuSigmaDist{c})
          case 'MvnInvWishartDist'
            mu = priorMuSigmaDist{c}.mu; T = priorMuSigmaDist{c}.Sigma; dof = priorMuSigmaDist{c}.dof; k = priorMuSigmaDist{c}.k;
            marginalDist(:,c) = copy(MvtDist(dof - 1 + 1, mu, T*(k+1)/(k*(dof-d+1))), nobs, 1);
          case 'MvnInvGammaDist'
            a = priorMuSigmaDist{c}.a; b = priorMuSigmaDist{c}.b; m = priorMuSigmaDist{c}.mu; k = priorMuSigmaDist{c}.k;
            marginalDist(:,c) = copy(StudentDist(2*a, m, b.*(1+k)./a), nobs, 1);
        end
      end

      % This is where the interesting stuff happend
      for itr=1:Nsamples
        if(verbose)
          fprintf('%d, ', itr)
        end
        for obs=1:nobs
          % For each iteration and observation, get the current assignment of the current observation
          % update the posterior parameters for its current cluster to reflect the absence of this observation
          kobs = curlatent(obs);
          xi = data(obs,:);
          prob = zeros(1,K);
          for k = 1:K
              [k0,v0,S0,m0] = updatePost(model,priorMuSigmaDist{k}, SSxbar, SSn, SSXX, SSXX2, xi,kobs);
              kn = k0 + 1;
              mn = (k0*colvec(m0) + xi')/kn;
              % Then, depending on the covariance type, update the marginal posterior distribution for this observation
              % to include this observation
              % Then, depending on the covariance type, update the marginal posterior distribution for this observation
              % to include this observation
              switch covtype{k}
                case 'full'
                  vn = v0 + 1;
                  Sn = S0 + xi'*xi + k0/(k0+1)*(xi'-colvec(m0))*(xi'-colvec(m0))';
                  marginalDist{obs,k} = setParamsAlt(marginalDist{obs,k}, vn - d + 1, mu, Sn);
                case 'spherical'
                  an = v0 + d/2;
                  bn = S0 + 1/2*sum(diag( xi'*x' + k0/(k0+1)*(xi'-colvec(m0))*(xi'-colvec(m0))' ));
                  marginalDist{obs,k} = setParamsAlt(marginalDist{obs,k}, 2*an, mn, bn.*(1+kn)./an);
                case 'diagonal'
                  an = v0 + 1/2;
                  bn = diag( S0*eye(d) + 1/2*(xi'*xi + k0/(k0+1)*(xi'-colvec(m0))*(xi'-colvec(m0))'));
                  marginalDist{obs,k} = setParamsAlt(marginalDist{obs,k}, 2*an, mn, bn.*(1+kn)./an);
              end
            lognij = log(SSn(k));
            % Now find the probability of this observation being in each cluster
            logmprob = logprob(marginalDist{obs,k}, xi);
            prob(k) =  lognij + logmprob;
          end % for clust = 1:K
          % normalize, sample, and then update the sufficient statistics to reflect the new assignment
          prob = colvec(exp(normalizeLogspace(prob)));
          curlatent(obs) = sample(prob,1);
          [SSxbar, SSn, SSXX, SSXX2] = SSaddXi(model, covtype{curlatent(obs)}, SSxbar, SSn, SSXX, SSXX2, curlatent(obs), data(obs,:));
        end % obs=1:n
        % Store the results if past burnin and if thinning permits
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

      mixDist = model.mixingWeights;

      musamples = zeros(outitr, d, K);
      Sigmasamples = zeros(outitr, d*d, K);

      mcmc.loglik = zeros(1,outitr);
      mix = zeros(outitr,K);
      tmpdistrib = model.distributions;
      tmpmodel = model;
      for itr=1:outitr
        mixFit = fit( mixDist, 'data', latent(itr,:)' );
        mix(itr,:) = rowvec(mixFit.T);
        for k=1:K
          [mu, Sigma, domain] = convertToMvn( fit(model.distributions{k}, 'data', data(latent(itr,:)' == k,:) ) );
          musamples(itr,:,k) = mu;
          Sigmasamples(itr,:,k) = rowvec(cholcov(Sigma));
        end
        %mcmc.loglik(:,itr) = logprobGibbs(tmpmodel,data,latent(:,itr));
      end
    muDist = SampleDist(musamples, 1:d);
    SigmaDist = SampleDist(Sigmasamples);
    mixDist = SampleDistDiscrete(mix, 1:K);
    end


    function [muoutDist, SigmaoutDist, latentoutDist, mixoutDist, permOut] = processLabelSwitch(model, latentDist, muDist, SigmaDist, mixDist, X, varargin)
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
      N = nsamples(latentDist);
      [n,d] = size(X);
      K = ndimensions(mixDist);

      % locally cache the samples
      latent = getsamples(latentDist);
      mix = getsamples(mixDist);
      mu = getsamples(muDist);
      Sigmatmp = getsamples(SigmaDist);

      % Need to post-process Sigma
      Sigma = zeros(d,d,N);
      for s=1:N
        for k=1:K
          Sigma(:,:,s,k) = reshape(Sigmatmp(s,:,k)',d,d)'*reshape(Sigmatmp(s,:,k)',d,d);
        end
      end

      % perm will contain the permutation that minimizes step two of the algorithm
      % oldPerm is the permutation that 
      % The permutations are as indices, is perm(1,1) indicates how we permute label 1 for iteration 1
      perm = bsxfun(@times,1:K,ones(N,K));
      oldPerm = bsxfun(@times,-inf,ones(N,K));
      fixedPoint = false;
      % For tracking purposes; value is how many times we have done the algorithm (itr and k already taken)
      klscore = 0;
      value = 1;
      fprintf('Attempting to resolve label switching ')
      while( ~fixedPoint )
        fprintf('...  ')
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
          if(verbose && mod(itr,500) == 0),fprintf('%d, ', itr); end;
          logqRik = zeros(n,K);
          for k=1:K
            % Probably just as best if we simply call mvnpdf here since we now know that we are working with MVN's
            logqRik(:,k) = log(mix(itr,oldPerm(itr,k))+eps)+ logprobMuSigma( model.distributions{k}, X, mu(itr,:,oldPerm(itr,k)), Sigma(:,:,itr,oldPerm(itr,k)) );
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
            % Probably just as best if we simply call mvnpdf here since we now know that we are working with MVN's
            logpij(:,k) = log(mix(itr,oldPerm(itr,k))+eps)+ logprobMuSigma( model.distributions{k}, X, mu(itr,:,oldPerm(itr,k)), Sigma(:,:,itr,oldPerm(itr,k)) );
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
      Sigmasamples = zeros(N,d*d,K);
      for itr=1:N
        latent(itr,:) = permOut(itr,latent(itr,:));
        mix(itr,:) = mix(itr,permOut(itr,:));
        for k=1:K
          mu(itr,:,k) = mu(itr,:,permOut(itr,k));
          Sigma(:,:,itr,k) = Sigma(:,:,itr,permOut(itr,k));
          Sigmasamples(itr,:,k) = rowvec(cholcov(Sigma(:,:,itr,k)));
        end
      end
      latentoutDist = SampleDistDiscrete(latent, 1:K);
      mixoutDist = SampleDistDiscrete(mix, 1:K);
      muoutDist = SampleDist(mu, 1:d);
      SigmaoutDist = SampleDist(Sigmasamples);  
      fprintf('\n')
    end

  end

  methods(Access = 'protected')

    function [model, prior, priorlik] = initializeGibbs(model,X)
      % we initialize by partitioning the observations into the K mixture components using kmeans
      % we return (initialized) priors for each model
      K = numel(model.distributions);
      [n,d] = size(X);
      %initassign = kmeans(X,K);
      prior = cell(K,1);
      priorlik = cell(K,1);
      for k=1:K
        switch class(model.distributions{k}.prior)
          case 'char'
            prior{k} = mkPrior(model.distributions{k},'data', X);
          otherwise
            prior{k} = model.distributions{k}.prior;
        end
            [model.distributions{k}.mu, model.distributions{k}.Sigma] = sample(prior{k});
            priorlik{k} = MvnConjugate(model.distributions{k}, 'prior', prior{k});
      end
    end

    function displayProgress(model,data,loglik,rr)
      figure(1000);
      clf
      t = sprintf('RR: %d, negloglik: %g\n',rr,-loglik);
      fprintf(t);
      if(size(data,2) == 2)
        nmixtures = numel(model.distributions);
        if(nmixtures == 2)
          colors = subd(predict(model,data),'T')';
          scatter(data(:,1),data(:,2),18,[colors(:,1),zeros(size(colors,1),1),colors(:,2)],'filled');
        else
          plot(data(:,1),data(:,2),'.','MarkerSize',10);
        end
        title(t);
        hold on;
        axis tight;
        for k=1:nmixtures
          f = @(x)sub(mean(model.mixingWeights),k)*exp(logprob(model.distributions{k},x));
          [x1,x2] = meshgrid(min(data(:,1)):0.1:max(data(:,1)),min(data(:,2)):0.1:max(data(:,2)));
          z = f([x1(:),x2(:)]);
          contour(x1,x2,reshape(z,size(x1)));
          mu = model.distributions{k}.mu;
          plot(mu(1),mu(2),'rx','MarkerSize',15,'LineWidth',2);
        end

      end
    end

    function [latent, prior, priorlik, SSxbar, SSn, SSXX, SSXX2] = initializeCollapsedGibbs(model,X)
      % latent is the column vector of intial assignments
      % prior{k} is the prior for distribution k
      % rest are sufficient statistics in the form of matrices (for more efficient computation)
      [n,d] = size(X);
      K = numel(model.distributions);
      latent = unidrnd(K,n,1);
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

    function [SSxbar, SSn, SSXX, SSXX2] = SSaddXi(model, covtype, SSxbar, SSn, SSXX, SSXX2, knew, xi)
      % Add contribution of xi to SS for cluster knew -- needed after sampling latent for each observation
      SSxbar(:,knew) = (SSn(knew)*SSxbar(:,knew) + xi')/(SSn(knew) + 1);
      switch lower(covtype)
        case 'diagonal'
          SSXX2(:,:,knew) = diag(diag( (SSn(knew) * SSXX2(:,:,knew) + xi'*xi) )) / (SSn(knew) + 1);
          SSXX(:,:,knew) = diag(diag( (SSn(knew) * SSXX2(:,:,knew) + xi'*xi - (SSn(knew)+1)*SSxbar(:,knew)*SSxbar(:,knew)') )) / (SSn(knew) + 1);
        case 'spherical'
          SSXX2(:,:,knew) = diag(sum(diag( (SSn(knew)*d * SSXX2(:,:,knew) + xi'*xi) )))/ ((SSn(knew) + 1)*d);
          SSXX(:,:,knew) = diag(sum(diag( (SSn(knew)*d * SSXX2(:,:,knew) + xi'*xi - (SSn(knew)+1)*SSxbar(:,knew)*SSxbar(:,knew)') )))/ ((SSn(knew) + 1)*d);
        case 'full'
          SSXX2(:,:,knew) = (SSn(knew) * SSXX2(knew) + xi'*xi) / (SSn(knew) + 1);
          SSXX(:,:,knew) = (SSn(knew) * SSXX2(:,:,knew) + xi'*xi - (SSn(knew)+1)*SSxbar(:,knew)*SSxbar(:,knew)') / (SSn(knew) + 1);
      end
      SSn(knew) = SSn(knew) + 1;
    end

    function [kn,vn,Sn,mn] = updatePost(model, priorMuSigmaDist, SSxbar, SSn, SSXX, SSXX2, xi, kobs)
      % returns parameters reflecting the loss of xi from cluster kobs
        n = SSn(kobs); d = length(xi);
          xbar = (SSn(kobs)*SSxbar(:,kobs) - xi')/(SSn(kobs) - 1);
          switch class(priorMuSigmaDist)
            case 'MvnInvWishartDist'
              k0 = priorMuSigmaDist.k; m0 = priorMuSigmaDist.mu;
              S0 = priorMuSigmaDist.Sigma; v0 = priorMuSigmaDist.dof;
              kn = k0 + n - 1;
              vn = v0 + n - 1;
              Sn = S0 + n*SSXX(:,:,kobs) - xi'*xi+ (k0*(n-1))/(k0+n-1)*(xbar-colvec(m0))*(xbar-colvec(m0))';
              mn = (k0*colvec(m0) + n*xbar)/kn;
            case 'MvnInvGammaDist'
              k0 = priorMuSigmaDist.Sigma; m0 = priorMuSigmaDist.mu;
              S0 = priorMuSigmaDist.b; v0 = priorMuSigmaDist.a;
              switch lower(covtype)
                case 'spherical'
                  vn = v0 + n*d/2;
                  Sn = S0 + 1/2*sum(diag( n*SSXX(:,:,kobs) - xi'*xi + (k0*(n-1))/(k0+n-1)*(xbar-colvec(m0))*(xbar-colvec(m0))' ));
                case 'diagonal'
                  vn = v0 + n/2;
                  Sn = diag( S0*eye(d) + 1/2*(n*SSXX(:,:,kobs) -xi'*xi + (k0*(n-1))/(k0+n-1)*(xbar-colvec(m0))*(xbar-colvec(m0))'));
              end
          end
    end

  end

end
