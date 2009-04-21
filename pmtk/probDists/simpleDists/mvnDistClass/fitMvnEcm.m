function [model, loglikTrace] = fitMvnEcm(model, data, prior, varargin)
% Find MLE/MAP estimate of MVN when X has missing values, using ECM algorithm
% obj is of type MvnDist
% data is an n*d design matrix with NaN values
% prior is either 'none' or 'NIW' or an MvnInvWishartDist object
% If prior='NIW', we use vague default hyper-params

% Written by Cody Severinski and Kevin Murphy

[maxIter, opttol, verbose] = process_options(varargin, ...
  'maxIter', 100, 'tol', 1e-4, 'verbose', false); 

[n,d] = size(data);

dataMissing = isnan(data);	  
missingRows = any(dataMissing,2);
missingRows = find(missingRows == 1);  
X = data'; % it will be easier to work with column vectros
 
% Initialize params
mu = nanmean(data); mu = mu(:);
Sigma = diag(nanvar(data));

expVals = zeros(d,n);
expProd = zeros(d,d,n);

% If there is no missing data, then just plug-in -- ECM not needed
for i=setdiffPMTK(1:n,missingRows)
  expVals(:,i) = X(:,i);
  expProd(:,:,i) = X(:,i)*X(:,i)';
end

iter = 1;
converged = false;
currentLL = -inf;
 
% Extract hyper-params for MAP estimation
prior = mkPrior(model, 'data', data, 'prior', model.prior, 'covtype', model.covtype);
switch class(prior)
  case 'MvnInvWishartDist'
    kappa0 = prior.k; m0 = prior.mu; nu0 = prior.dof; T0 = prior.Sigma;
  case 'MvnInvGammaDist'
    kappa0 = prior.Sigma; m0 = prior.mu; nu0 = prior.a; T0 = prior.b;
  case 'NoPrior'
    % Setting hyperparams to zero gives the MLE
    kappa0 = 0; m0 = zeros(d,1); nu0 = 0; T0 = zeros(d,d);
  otherwise
    error(['unknown prior ' classname(prior)])
end

while(~converged)
		% Expectation step
    for i=missingRows(:)'
      u = dataMissing(i,:); % unobserved entries
      o = ~u; % observed entries
      Sooinv = inv(Sigma(o,o));
      Si = Sigma(u,u) - Sigma(u,o) * Sooinv * Sigma(o,u);
      expVals(u,i) = mu(u) + Sigma(u,o)*Sooinv*(X(o,i)-mu(o));
			% We never did actually update the actual expVals with the values of the observed dimensions
			expVals(o,i) = X(o,i);
      expProd(u,u,i) = expVals(u,i) * expVals(u,i)' + Si;
      expProd(o,o,i) = expVals(o,i) * expVals(o,i)';
      expProd(o,u,i) = expVals(o,i) * expVals(u,i)';
      expProd(u,o,i) = expVals(u,i) * expVals(o,i)';
    end

		%  M step 
		% we store the old values of mu, Sigma just in case the log likelihood decreased and we need to return the last values before the singularity occurred
		muOld = mu;
		SigmaOld = Sigma;

		% MAP estimate for mu -- note this reduces to the MLE if kappa0 = 0
    mu = (sum(expVals,2) + kappa0*m0)/(n + kappa0);
    % Compute ESS = 1/n sum_i E[ (x_i-mu) (x_i-mu)' ] using *new* value of mu
    % ESS = sum(expProd,3) + n*mu*mu' - 2*mu*sum(expVals,2)';
		ESS = sum(expProd,3) - sum(expVals,2)*mu' - mu*sum(expVals,2)' + n*mu*mu';

		switch class(prior)
			case 'char'
				switch lower(prior)
					case 'none'
						% If no prior, then we have that Sigma = 1/n* E[(x_i - mu)(x_i - mu)'] = ESS / n
						Sigma = ESS/n;
					case 'niw'
						Sigma = (ESS + T0 + kappa0*(mu-m0)*(mu-m0)')/(n+nu0+d+2);
					otherwise
						error(['unknown prior ' prior])
				end
			case 'MvnInvWishartDist'
					Sigma = (ESS + T0 + kappa0*(mu-m0)*(mu-m0)')/(n+nu0+d+2);
      case 'MvnInvGammaDist'
        switch lower(model.covtype)
          case 'diagonal'
            Sigma = diag(ESS + T0 + kappa0*(mu-mu0)*(mu-mu0)') / (n + nu0 + 1 + 2);
          case 'spherical'
            Sigma = diag(sum(diag(ESS + T0 + kappa0*(mu-mu0)*(mu-mu0)'))) / (n*d + nu0 + d + 2);
        end
			otherwise
					error(['unknown prior ' prior])
		end
		
		if(det(Sigma) <= 0)
			warning('fitMvnEcm:Sigma:nonsingular','Warning: Obtained Nonsingular Sigma.  Exiting with last reasonable parameters \n')
			mu = muOld;
			Sigma = SigmaOld;
			return;
		end
    
    % Convergence check
    prevLL = currentLL;
    %currentLL = -n/2*logdet(2*pi*Sigma) - 0.5*trace(SS.XX*prec);
    % SS.n
    % SS.xbar = 1/n sum_i X(i,:)'
    % SS.XX2(j,k) = 1/n sum_i X(i,j) X(i,k)
    %SS.XX2 = ESS/n; SS.n = n; SS.xbar = mu/n; % I don't think that SS.xbar = mu / n, but rather just mu;  Also, SS.XX2 looks wrong...
		%SS.XX2 = sum(expProd,3)/n; SS.n = n; SS.xbar = mu; % Actually, it looks like SS.xbar actually needs to be sum(expVals,2) / n;
		SS.XX2 = sum(expProd,3)/n; SS.n = n; SS.xbar = sum(expVals,2)/n;
    currentLL = logprobSS(MvnDist(mu,Sigma), SS);

    if isa(prior, 'MvnInvWishartDist')
      currentLL = currentLL + logprob(prior, mu, Sigma);
    end
    loglikTrace(iter) = currentLL;
    if (currentLL < prevLL)
			warning('fitMvnEcm:loglik:nonincrease','warning: EM did not increase objective.  Exiting with last reasonable parameters \n')
			mu = muOld;
			Sigma = SigmaOld;
		end
    if verbose, fprintf('%d: LL = %5.3f\n', iter, currentLL); end
    iter = iter + 1;
    converged = iter >=maxIter || (abs(currentLL - prevLL) / (abs(currentLL) + abs(prevLL) + eps)/2) < opttol;
end

 
