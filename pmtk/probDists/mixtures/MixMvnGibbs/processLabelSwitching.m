

    function [mcmc,permOut] = processLabelSwitch(model, latentDist, muDist, SigmaDist, mixDist, X, varargin)
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

      % Need to post-process the SigmaDist
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
            try
            logqRik(:,k) = log(mix(itr,oldPerm(itr,k))+eps)+ logprobMuSigma( model.distributions{k}, X, mu(itr,:,oldPerm(itr,k)), Sigma(:,:,itr,oldPerm(itr,k)) );
            catch ME
              keyboard
            end
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
            try
            logpij(:,k) = log(mix(itr,oldPerm(itr,k))+eps)+ logprobMuSigma( model.distributions{k}, X, mu(itr,:,oldPerm(itr,k)), Sigma(:,:,itr,oldPerm(itr,k)) );
            catch ME
              keyboard
            end
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
        latent(itr,:) = permOut(itr,latent(itr,:));
        mix(itr,:) = mix(itr,permOut(itr,:));
        for k=1:K
          mu(itr,:,k) = mu(itr,:,permOut(itr,:));
          Sigma(:,:,itr,k) = Sigma(:,:,itr,permOut(itr,k));
        end
      end

    end
