    function [muoutDist, SigmaoutDist, mixoutDist, latentoutDist] = processLabelSwitching(muDist, SigmaDist, mixDist, latentDist, X, varargin)
      % Implements the KL - algorithm for label switching from 
      %@article{ stephens2000dls,
      % title = "{Dealing with label switching in mixture models}",
      % author = "M. Stephens",
      % journal = "Journal of the Royal Statistical Society. Series B, Statistical Methodology",
      % pages = "795--809",
      % year = "2000",
      % publisher = "Blackwell Publishers"
      %}
      [verbose, maxitr] = processArgs(varargin, '-verbose', true, '-maxitr', inf);
      %muDist = dists.muDist;
      %SigmaDist = dists.SigmaDist;
      %mixDist = dists.mixDist;
      %latentDist = dists.latentDist;

      nSamples = size(latentDist.samples, 2);
      [nObs,d] = size(X);
      K = size(mixDist.samples,1);

      latent = latentDist.samples';
      mix = mixDist.samples';
      mu = zeros(nSamples,d,K);
      Sigma = zeros(d,d,nSamples,K);
      for k=1:K
        mu(:,:,k) = muDist{k}.samples';
        Sigma(:,:,:,k) = SigmaDist{k}.samples;
      end
      % Save lognormconst
      invS = zeros(d,d,nSamples,K);
      for s=1:nSamples
        for k=1:K
          invS(:,:,s,k) = inv(Sigma(:,:,s,k));
          logconst(s,k) = 1/2*logdet(2*pi*Sigma(:,:,s,k));
        end
      end

      % perm will contain the permutation that minimizes step two of the algorithm
      % The permutations are as indices, is perm(1,1) indicates how we permute label 1 for iteration 1
      perm = bsxfun(@times,1:K,ones(nSamples,K));
      ident = bsxfun(@times, 1:K, ones(nSamples,K));
      oldPerm = bsxfun(@times,-inf,ones(nSamples,K));
      fixedPoint = false;
      % For tracking purposes; value is how many times we have done the algorithm (itr and k already taken)
      klscore = 0;
      value = 1;
      fail = false;
      fprintf('Attempting to resolve label switching \n')
      while( ~fixedPoint )
        if(verbose), fprintf('(Run %d). Computing Q for iteration  ', value);end;
        Q = zeros(nObs,K); logpdata = zeros(nObs,nSamples,K);
        %oldPerm = perm;
        % Note that doing both qmodel and pmodel in the same loop is more efficient in terms of runtime
        % but we need to store pij for each iteration.  This can cause memory to run out if 
        % either nobs or iter is large.
        % Hence, we first compute Q and then pij.
        for itr = 1:nSamples
          logqRik = zeros(nObs,K);
          for k=1:K
            %XC = bsxfun(@minus, X, mu(itr,:,oldPerm(itr,k)));
            %logpdata(:,itr,k) = -logconst(itr,oldPerm(itr,k)) - 1/2*sum((XC*invS(:,:,itr,oldPerm(itr,k))).*XC,2);
            %logqRik(:,k) = log(mix(itr,oldPerm(itr,k))+eps)+ logpdata(:,itr,k);
            XC = bsxfun(@minus, X, mu(itr,:,k));
            logpdata(:,itr,k) = -logconst(itr,k) - 1/2*sum((XC*invS(:,:,itr,k)).*XC,2);
            logqRik(:,k) = log(mix(itr,k))+ logpdata(:,itr,k);
          end
          Q = Q + exp(normalizeLogspace(logqRik));
        end
        Q = Q / nSamples;
        if(verbose)
          fprintf('computed.  Optimizing over permutations.  ')
        end
        % Loss for each individual iteration
        loss = zeros(nSamples,1);
        for itr = 1:nSamples
          logpij = zeros(nObs,K);
          for k=1:K
            %logpij(:,k) = log(mix(itr,oldPerm(itr,k))+eps) + logpdata(:,itr,oldPerm(itr,k));
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
        if(any(perm(itr,:) == 0)), keyboard, end;
        end
        % KL loss is the sum of all the losses over all the iterations
        klscore(value) = mean(loss);

        if(value == 1)
          deltakl = 0;
        else
          deltakl = klscore(value) - klscore(value - 1);
        end
        if(verbose),fprintf('KL Loss = %g.  Delta = %g \n',klscore(value), deltakl); end;
        % Stopping criteria - what would be ideal is to have a vector of stopping criteria
        % and have the user select the stopping criteria
        % I'm thinking that we could pass this in as varargin, and then evaluate the chosen
        % criteria after each run
        %if( value > 2 && (all(all(perm == oldPerm)) || approxeq(klscore(value), klscore(value-1), 1/klscore(value), 1) || approxeq(klscore(value), klscore(value-2), 1/klscore(value), 1) ) )
        %if(value > 2 && (all(all(perm == ident)) || value >= maxitr || convergenceTest(klscore(value), klscore(value - 1))))
        if(value > 2 && (all(all(perm == oldPerm)) || value >= maxitr || convergenceTest(klscore(value), klscore(value - 1))))
          fixedPoint = true;
        end
        if(deltakl > 0)
          warning('Objective did not decrease.  Returning with last good permutation');
          permOut = oldPerm;
          fixedPoint = true;
          fail = true;
        end

        if(fail ~= true && ~fixedPoint)
          for k=1:K
            mu(itr,:,k) = mu(itr,:,perm(itr,k));
            logconst(itr,k) = logconst(itr,perm(itr,k));
            invS(:,:,itr,k) = invS(:,:,itr,perm(itr,k));
          end
        end
        value = value + 1;   
      end
      permOut = perm;
      latentout = zeros(nSamples,nObs);
      mixout = zeros(nSamples,K);
      muout = zeros(nSamples,d,K);
      Sigmaout = zeros(d,d,nSamples,K);
      for itr=1:nSamples
        latentout(itr,:) = permOut(itr,latent(itr,:));
        mixout(itr,:) = mix(itr,permOut(itr,:));
        for k=1:K
          muout(itr,:,k) = mu(itr,:,permOut(itr,k));
          Sigmaout(:,:,itr,k) = Sigma(:,:,itr,permOut(itr,k));
        end
      end
      latentoutDist = SampleDist(latentout');
      mixoutDist = SampleDist(mixout');
      muoutDist = cell(K,1); SigmaoutDist = cell(K,1);
      for k=1:K
        muoutDist{k} = SampleDist(muout(:,:,k)');
        SigmaoutDist{k} = SampleDist(Sigmaout(:,:,:,k)); 
      end 
      distsout = struct;%('muDist', muoutDist, 'SigmaDist', SigmaoutDist, 'mixDist', mixoutDist, 'latentDist', latentoutDist);
      distsout.muDist = muoutDist;
      distsout.SigmaDist = SigmaoutDist;
      distsout.mixoutDist = mixoutDist;
      distsout.latentDist = latentoutDist;
      fprintf('\n')
    end
