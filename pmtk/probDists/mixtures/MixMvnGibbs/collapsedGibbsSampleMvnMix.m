
    function mcmc = collapsedGibbsSampleMvnMix(model,data,varargin)
      % Collapsed Gibbs sampling for a mixture of MVNs
      [Nsamples, Nburnin, thin, verbose] = process_options(varargin, ...
        'Nsamples'  , 1000, ...
        'Nburnin'   , 500, ...
        'thin'    , 1, ...
        'verbose'   , false );
      [nobs,d] = size(data);
      K = numel(model.distributions);
      outitr = ceil((Nsamples - Nburnin) / thin);
      latent = zeros(nobs,outitr);
      keep = 1;

      if(verbose)
        fprintf('Initializing Gibbs sampler... \n')
      end
      [latent(:,1), priorMuSigmaDist, SSxbar, SSn, SSXX, SSXX2] = initializeCollapsedGibbs(model,data);
      curlatent = latent(:,1);
      % marginalDist stores objects for each observation, representing the marginal probabiltiy of each observation
      % being part of each cluster
      marginalDist = cell(nobs,K);
      % covtype will store the covariance structure for each cluster, since these need not be identical a priori
      covtype = cell(1,K);

      for c=1:K
        covtype{c} = model.distributions{c}.covtype;
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
        if(itr > Nburnin && mod(itr, thin)==0)
          latent(:,keep) = curlatent;
          keep = keep + 1;
        end
      end
      if(verbose)
        fprintf('\n')
      end

      % With Gibbs sampling complete, create the mcmc struct needed for returning
      mcmc.latent = latent;
      mcmc.loglik = zeros(1,outitr);
      mcmc.mix = zeros(K,outitr);
      mcmc.param = cell(K,outitr);
      param = cell(K,1);
      tmpdistrib = model.distributions;
      tmpmodel = model;
      for itr=1:outitr
        latent = mcmc.latent;
        mix = histc(latent(:,itr), 1:K) / nobs;
        for k=1:K
          [mu, Sigma, domain] = convertToMvn( fit(model.distributions{k}, 'data', data(latent(:,itr) == k,:) ) );
          param{k} = struct('mu', mu, 'Sigma', Sigma);
          tmpdistrib{k} = setParamsAlt(tmpdistrib{k}, mu, Sigma);
        end
        mcmc.param(:,itr) = param;
        tmpmodel = setParamsAlt(tmpmodel, mix, tmpdistrib);
        mcmc.mix(:,itr) = mix;
        mcmc.loglik(:,itr) = logprobGibbs(tmpmodel,data,latent(:,itr));
      end
    end

 function [latent, prior, SSxbar, SSn, SSXX, SSXX2] = initializeCollapsedGibbs(model,X)
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
        prior{k} = mkPrior(model.distributions{k},X);
        SS{k} = mkSuffStat(model.distributions{k}, X(latent == k,:));
        SSn(k) = SS{k}.n;
        SSxbar(:,k) = SS{k}.xbar;
        SSXX(:,:,k) = SS{k}.XX;
        SSXX2(:,:,k) = SS{k}.XX2;
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
