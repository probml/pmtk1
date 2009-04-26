function [distributions, bestMix, loglikTrace, itr] = EMforGMM(distributions, mixDistrib, X, varargin)

  [verbose, maxItr, tol, nrestarts] = processArgs(varargin, '-verbose', true, '-maxItr', 50, '-tol', 1e-03, '-nrestarts', 1);

  K = numel(distributions);
  [n,d] = size(X);
  mixDistrib = initPrior(mixDistrib);
  alpha = mixDistrib.prior.alpha;
  mixPrior = mixDistrib.prior;
  mu = zeros(K,d);
  Sigma = zeros(d,d,K);
  mix = zeros(K,1);
  prior = cell(K,1);
  logpriordist = zeros(K,1);
  covtype = cell(K,1);

  m0 = zeros(K,d);
  k0 = zeros(K,1);
  T0 = zeros(d,d,K);
  v0 = zeros(K,1);

  for k=1:K
    %mu(k,:) = distributions{k}.mu';
    %covar(:,:,k) = distributions{k}.Sigma;

    covtype{k} = distributions{k}.covtype;
    prior{k} = mkPrior(distributions{k},'-data', X);
    switch class(prior{k})
      case 'MvnInvWishartDist'
        % The next lines cannot be factored out
        % as it might be a NoPrior
        m0(k,:) = prior{k}.mu';
        T0(:,:,k) = prior{k}.Sigma;
        k0(k) = prior{k}.k;
        v0(k) = prior{k}.dof;
      case 'MvnInvGammaDist'
        m0(k,:) = prior{k}.mu';
        T0(:,:,k) = prior{k}.b;
        k0(k) = prior{k}.Sigma;
        v0(k) = prior{k}.a;
      otherwise
      % We do nothing;
    end %switch class prior;
  end
  mix = mixDistrib.T;
  bestLL = -inf;
  for restart = 1:nrestarts

    % Initialize
    [mu, assign] = kmeansSimple(X, K);
    v = var(X(:));
    for k=1:K
      C = cov(X(assign == k,:));
      Sigma(:,:,k) = C + (v/K)*eye(d); % diag(diag(C));
      %Sigma(:,:,k) = cov(X(assign == k,:)) + 0.1*eye(d);
      %Sigma(:,:,k) = cov(X) / K;
      switch class(prior{k})
        case 'NoPrior'
          logpriordist(k) = 0;
        case 'MvnInvWishartDist'
          logpriordist(k) = - 1/2*logdet(2*pi*Sigma(:,:,k)/k0(k)) ...
                            - k0(k)/2*(mu(k,:) - m0(k,:))*inv(Sigma(:,:,k))*(mu(k,:) - m0(k,:))' ...
                            - 1/2*trace(inv(T0(:,:,k)*Sigma(:,:,k))) ...
                            - (v0(k)+d+1)/2*logdet(Sigma(:,:,k)) ...
                            - (v0(k)*d/2)*log(2) + mvtGammaln(d,v0(k)/2) -(v0(k)/2)*logdet(T0(:,:,k));
        case 'MvnInvGammaDist'
          T0 = rowvec(diag(T0(:,:,k)));
          logpriordist(k) = - 1/2*logdet(2*pi*Sigma(:,:,k)/k0(k)) ...
                            - k0(k)/2*(mu(k,:) - m0(k,:))*inv(Sigma(:,:,k))*(mu(i,:) - m0(k,:))' ...
                            - sum(T0.*log(v0(k)) - gammaln(v0(k)) ...
                            - (v0(k)+1).*rowvec(diag(Sigma(:,:,k))) - T0./rowvec(diag(Sigma(:,:,k))));
      end
    end
    logpriormix = gammaln(sum(alpha)) - sum(gammaln(alpha)) + sum((alpha - 1).*log(mix));
    logprior = sum(logpriordist) + logpriormix;
    mix = normalize(histc(assign, 1:K) + alpha - 1);
    % End initialization

    converged = false; itr = 0; 
    currentLL = -inf;
    loglikTrace = [];
    while(not(converged))
      prevLL = currentLL;
      % E step
      logpij = zeros(n,K);
      for k=1:K
        XC = bsxfun(@minus, X, mu(k,:));
        logpij(:,k) = -1/2*(logdet(2*pi*Sigma(:,:,k)) + sum((XC*inv(Sigma(:,:,k))).*XC,2));
      end
      [logpij, LL] = normalizeLogspace(logpij);
      pij = exp(logpij);
      L = sum(LL); L = L + logprior; currentLL = L / n;
      [EMn, EMxbar, EMXX, EMXX2, EMcounts] = emSS(X, K, pij);
      % End E step

      % M step
      for k=1:K
        kn = k0(k) + EMn(k);
        mn = (k0(k)*m0(k,:) + EMn(k)*EMxbar(k,:)) / kn;
        switch class(prior{k})
          case 'NoPrior'
            Sigma(:,:,k) = EMXX(:,:,k);
          case 'MvnInvWishartDist'
            vn = v0(k) + EMn(k);
            Sn = T0(:,:,k) + EMn(k)*EMXX(:,:,k) + k0(k)*EMn(k)/(k0(k) + EMn(k)) *(EMxbar(k,:) - m0(k,:))'*(EMxbar(k,:) - m0(k,:));

            Sigma(:,:,k) = Sn / (vn + d + 2);
          case 'MvnInvGammaDist'
            switch lower(covtype)
              case 'spherical'
                vn = v0(k) + EMn(k)*d/2;
                Sn = diag(T0(:,:,k) + 1/2*( EMn(k)*EMXX(:,:,k) + k0(k)*EMn(k) / (k0(k) + EMn(k)) * (EMxbar(k,:) - m0(k,:))'*(EMxbar(k,:) - m0(k,:)) ) );

                Sigma(:,:,k) = diag(Sn ./ (vn + d/2 + 1));
              
              case 'diagonal'
                vn = v0(k) + EMn(k)/2;
                Sn = diag( diag(b0) + 1/2*(EMn(k)*EMXX(:,:,k) + k0(k)*EMn(k) / (k0(k) + EMn(k)) *(EMxbar(k,:) - m0(k,:))'*(EMxbar(k,:) - m0(k,:))) );

                Sigma(:,:,k) = diag( Sn ./ (vn + 1/2 + 1));
            end % lower(covtype)
        end % class(prior)
        mu(k,:) = mn;
      end % for k=1:K
      mix = normalize(EMcounts + alpha - 1);

      % M step
      itr = itr + 1;
      if (verbose), displayProgress(currentLL,itr,restart); end;
      if(itr > 2),converged = itr >= maxItr || convergenceTest(currentLL, prevLL, tol); end;
      if(currentLL < prevLL)
        warning(sprintf('\n EM not monotonically increasing objective (delta = %g)', currentLL -prevLL))
      end;
      loglikTrace(itr) = currentLL;
    end % while(not(converged))

    % save the parameters if we did better in this iteration
    if(currentLL > bestLL)
      bestMu = mu;
      bestSigma = Sigma;
      bestMix = mix;
      bestLL = currentLL;
    end
    % End save the parameters

  end % restart = 1:nrestarts

  % return the best model
  for k=1:K
    distributions{k}.mu = bestMu(k,:)';
    distributions{k}.Sigma = bestSigma(:,:,k);
  end
  mixingDistrib.T = bestMix;
  % End return the best model

end % of EMforGMM

function [SSn, SSxbar, SSXX, SSXX2, counts] = emSS(X, K, weights)
  % This is basically a specialized version of MvnDist().mkSuffStat for our purposes
  SSn = sum(weights,1);
  for k=1:K
    SSxbar(k,:) = sum(bsxfun(@times, X, weights(:,k))) / SSn(k);
    SSXX2(:,:,k) = bsxfun(@times, X, weights(:,k))'*X / SSn(k);
    XC = bsxfun(@minus,X,SSxbar(k,:));
    SSXX(:,:,k) = bsxfun(@times, XC, weights(:,k))'*XC / SSn(k); 
  end
  counts = colvec(normalize(sum(weights,1)));
end

function displayProgress(loglik,iter,r) %#ok that ignores model
  % override in subclass with more informative display
  t = sprintf('EM restart %d iter %d, negloglik %g\n',r,iter,-loglik);
  fprintf(t);
end
