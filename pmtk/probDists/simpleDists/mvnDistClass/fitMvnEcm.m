function model = fitMvnEcm(model, data, prior)
% Find MLE/MAP estimate of MVN when X has missing values, using ECM algorithm
% obj is of type MvnDist
% data is an n*d design matrix with NaN values
% prior is either 'none' or an MvnInvWishartDist object

% Based on code by Cody Severinski, modified by Kevin Murphy

maxIter = 100;
opttol = 1e-3;
[n,d] = size(data);

dataMissing = isnan(data);	  
missingRows = any(dataMissing,2);
missingRows = find(missingRows == 1);  
X = data'; % it will be easier to work with column vectros
 
% Initialize params
mu = nanmean(data);
Sigma = diag(nanvar(data));

expVals = zeros(d,n);
expProd = zeros(d,d,n);

for i=mysetdiff(1:n,missingCases)
  expVals(:,i) = X(:,i);
  expProd(:,:,i) = X(:,i)*X(:,i)';
end
oldloglikelihood = -inf;
newloglikelihood = +inf;
loglikelihood = zeros(1,maxitr+1);
iter = 1;
converged = false;
currentLL = -inf;
 
% Extract hyper-params for MAP estimation
switch class(prior)
  case 'char'
    switch prior
      case 'none'
        kappa0 = 0; m0 = zeros(d,1);
        nu0 = 0; T0 = zeros(d,d);
      otherwise
        error(['unknown prior ' prior])
    end
  case 'MvnInvWishartDist'
    kappa0 = prior.k; m0 = prior.mu;
    nu0 = prior.dof; T0 = prior.Sigma;
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
    % First estimate mu 
    mu = (sum(expVals,2) + kappa0*m0)/(n + kappa0);
    % Compute ESS = 1/n sum_i E[ (x_i-mu) (x_i-mu)' ] using *new* value of mu
    ESS = sum(expProd,3) + n*mu*mu' - 2*mu*sum(expVals,2)';
    % Compute Sigma
    Sigma = (ESS + T0 + kappa0*(mu-mo)*(mu-mo)')/(n+nu0+d+2);
    
    
    % Convergence check
    prevLL = currentLL;
    prec = inv(Sigma);
    currentLL = -n/2*logdet(2*pi*Sigma) - 0.5*trace(SS.XX*prec);
    if isa(prior, 'MvnInvWishartDist')
      currentLL = current + logprob(prior, mu, Sigma);
    end
    iter = iter + 1;
    converged = iter >=maxiter || (abs(currentLL - prevLL) / (abs(currentLL) + abs(prevLL) + eps)/2) < opttol;
end

 
