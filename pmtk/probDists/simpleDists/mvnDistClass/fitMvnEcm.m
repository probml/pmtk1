function [model, loglikTrace] = fitMvnEcm(model, data, prior, varargin)
% Find MLE/MAP estimate of MVN when X has missing values, using ECM algorithm
% obj is of type MvnDist
% data is an n*d design matrix with NaN values
% prior is either 'none' or 'NIW' or an MvnInvWishartDist object
% If prior='NIW', we use vague default hyper-params

% Based on code by Cody Severinski, modified by Kevin Murphy

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

for i=mysetdiff(1:n,missingRows)
  expVals(:,i) = X(:,i);
  expProd(:,:,i) = X(:,i)*X(:,i)';
end
iter = 1;
converged = false;
currentLL = -inf;
 
% Extract hyper-params for MAP estimation
switch class(prior)
  case 'char'
    switch lower(prior)
      case 'none'
        % Setting hyperparams to zero gives the MLE
        kappa0 = 0; m0 = zeros(d,1);
        nu0 = 0; T0 = zeros(d,d);
      case 'niw'
        prior = MvnDist.mkNiwPrior(data);
        kappa0 = prior.k; m0 = prior.mu; nu0 = prior.dof; T0 = prior.Sigma;
      otherwise
        error(['unknown prior ' prior])
    end
  case 'MvnInvWishartDist'
    kappa0 = prior.k; m0 = prior.mu; nu0 = prior.dof; T0 = prior.Sigma;
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
      expProd(u,u,i) = expVals(u,i) * expVals(u,i)' + Si;
      expProd(o,o,i) = expVals(o,i) * expVals(o,i)';
      expProd(o,u,i) = expVals(o,i) * expVals(u,i)';
      expProd(u,o,i) = expVals(u,i) * expVals(o,i)';
    end

		%  M step 
    mu = (sum(expVals,2) + kappa0*m0)/(n + kappa0);
    % Compute ESS = 1/n sum_i E[ (x_i-mu) (x_i-mu)' ] using *new* value of mu
    ESS = sum(expProd,3) + n*mu*mu' - 2*mu*sum(expVals,2)';
    Sigma = (ESS + T0 + kappa0*(mu-m0)*(mu-m0)')/(n+nu0+d+2);
    
    
    % Convergence check
    prevLL = currentLL;
    %currentLL = -n/2*logdet(2*pi*Sigma) - 0.5*trace(SS.XX*prec);
    % SS.n
    % SS.xbar = 1/n sum_i X(i,:)'
    % SS.XX2(j,k) = 1/n sum_i X(i,j) X(i,k)
    SS.XX2 = ESS/n; SS.n = n; SS.xbar = mu/n;
    currentLL = logprobSS(MvnDist(mu,Sigma), SS);
    if isa(prior, 'MvnInvWishartDist')
      currentLL = currentLL + logprob(prior, mu, Sigma);
    end
    loglikTrace(iter) = currentLL;
    if currentLL < prevLL, sprintf('warning: EM did not increase objective'); end
    if verbose, fprintf('%d: LL = %5.3f\n', iter, currentLL); end
    iter = iter + 1;
    converged = iter >=maxIter || (abs(currentLL - prevLL) / (abs(currentLL) + abs(prevLL) + eps)/2) < opttol;
end

 
