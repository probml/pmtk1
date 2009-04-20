    function [distsout, permOut] = processLabelSwitching(dists, X, varargin)
      % Implements the KL - algorithm for label switching from 
      %@article{ stephens2000dls,
      % title = "{Dealing with label switching in mixture models}",
      % author = "M. Stephens",
      % journal = "Journal of the Royal Statistical Society. Series B, Statistical Methodology",
      % pages = "795--809",
      % year = "2000",
      % publisher = "Blackwell Publishers"
      %}
      [verbose, stopCriteria, maxitr] = process_options(varargin, 'verbose', true, 'stopCriteria', 1, 'maxitr', inf);
      muDist = dists.muDist;
      SigmaDist = dists.SigmaDist;
      mixDist = dists.mixDist;
      latentDist = dists.latentDist;

      N = nsamples(latentDist);
      [n,d] = size(X);
      K = ndimensions(mixDist);

      % locally cache the samples
      latent = getsamples(latentDist);
      mix = getsamples(mixDist);
      mu = getsamples(muDist);
      Sigmatmp = getsamples(SigmaDist);

      % Need to post-process Sigma
      Sigma = zeros(d,d,N,K); invS = zeros(d,d,N,K);
      for s=1:N
        for k=1:K
          Sigma(:,:,s,k) = reshape(Sigmatmp(s,:,k)',d,d)'*reshape(Sigmatmp(s,:,k)',d,d);
          invS(:,:,s,k) = inv(Sigma(:,:,s,k));
          logconst(s,k) = 1/2*logdet(2*pi*Sigma(:,:,s,k));
        end
      end

      % perm will contain the permutation that minimizes step two of the algorithm
      % The permutations are as indices, is perm(1,1) indicates how we permute label 1 for iteration 1
      perm = bsxfun(@times,1:K,ones(N,K));
      oldPerm = bsxfun(@times,-inf,ones(N,K));
      fixedPoint = false;
      % For tracking purposes; value is how many times we have done the algorithm (itr and k already taken)
      klscore = 0;
      value = 1;
      fprintf('Attempting to resolve label switching \n')
      while( ~fixedPoint )
        if(verbose), fprintf('(Run %d). Computing Q for iteration  ', value);end;
        Q = zeros(n,K); logpdata = zeros(n,N,K);
        %oldPerm = perm;
        % Note that doing both qmodel and pmodel in the same loop is more efficient in terms of runtime
        % but we need to store pij for each iteration.  This can cause memory to run out if 
        % either nobs or iter is large.
        % Hence, we first compute Q and then pij.
        for itr = 1:N
          logqRik = zeros(n,K);
          for k=1:K
            %XC = bsxfun(@minus, X, mu(itr,:,oldPerm(itr,k)));
            XC = bsxfun(@minus, X, mu(itr,:,k));
            %logpdata(:,itr,k) = -logconst(itr,oldPerm(itr,k)) - 1/2*sum((XC*invS(:,:,itr,oldPerm(itr,k))).*XC,2);
            logpdata(:,itr,k) = -logconst(itr,k) - 1/2*sum((XC*invS(:,:,itr,k)).*XC,2);
            %logqRik(:,k) = log(mix(itr,oldPerm(itr,k))+eps)+ logpdata(:,itr,k);
            logqRik(:,k) = log(mix(itr,k))+ logpdata(:,itr,k);
          end
          Q = Q + exp(normalizeLogspace(logqRik));
        end
        Q = Q / N;
        if(verbose)
          fprintf('computed.  Optimizing over permutations.  ')
        end
        % Loss for each individual iteration
        loss = zeros(N,1);
        for itr = 1:N
          logpij = zeros(n,K);
          for k=1:K
            %logpij(:,k) = log(mix(itr,oldPerm(itr,k))+eps) + logpdata(:,itr,k);
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
        for k=1:K
          mu(itr,:,k) = mu(itr,:,perm(itr,k));
          logconst(itr,k) = logconst(itr,perm(itr,k));
          invS(:,:,itr,k) = invS(:,:,itr,perm(itr,k));
        end
        end
        % KL loss is the sum of all the losses over all the iterations
        klscore(value) = mean(loss);

        % Stopping criteria - what would be ideal is to have a vector of stopping criteria
        % and have the user select the stopping criteria
        % I'm thinking that we could pass this in as varargin, and then evaluate the chosen
        % criteria after each run
        %if( value > 2 && (all(all(perm == oldPerm)) || approxeq(klscore(value), klscore(value-1), 1/klscore(value), 1) || approxeq(klscore(value), klscore(value-2), 1/klscore(value), 1) ) )
        if(all(all(perm == oldPerm)) || value >= maxitr)
          fixedPoint = true;
        end
        oldPerm = perm;
        if(value == 1)
          deltakl = 0;
        else
          deltakl = klscore(value) - klscore(value - 1);
        end
        if(verbose),fprintf('KL Loss = %g.  Delta = %g \n',klscore(value), deltakl); end;
        value = value + 1;   
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
      distsout = struct('muDist', muoutDist, 'SigmaDist', SigmaoutDist, 'mixDist', mixoutDist, 'latentDist', latentoutDist);
      fprintf('\n')
    end
