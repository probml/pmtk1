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
  expValues(:,i) = X(:,i);
  expProduct(:,:,i) = X(:,i)*X(:,i)';
end
oldloglikelihood = -inf;
newloglikelihood = +inf;
loglikelihood = zeros(1,maxitr+1);
iter = 1;
converged = false;
currentLL = -inf;
 
while(~converged)
		% Expectation step
    for i=missingRows(:)'
      u = dataMissing(i,:); % unobserved entries
      o = ~u; % observed entries
      Sooinv = inv(Sigma(o,o,));
      Si = Sigma(u,u) - Sigma(u,o) * Sooinv * Sigma(o,u);
      expVals(u,i) = mu(u) + Sigma(u,o)*Sooinv*(X(o,i)-mu(o));
      expProd(u,u,i) = expVals(u,i) * expVals(u,i)' + Si;
      expProd(o,o,i) = expVals(o,i) * expVals(o,i)';
      expProd(o,u,i) = expVals(o,i) * expVals(u,i)';
      expProd(u,o,i) = expVals(u,i) * expVals(o,i)';
    end

		%  M step
    % First estimate mu using MLE
    mu = sum(expVals,2)/n;
    % Then compute SS = 1/n sum_i E[ (x_i-mu) (x_i-mu)' ] 
    %  = 1/n sum_i ( E[x_i x_i'] - 2 mu E[x_i]' + mu mu' ) 
    SS.XX = sum(expProd,3) - 2*mu*sum(expVals,2)' + mu*mu';
    SS.XX = SS.XX/n;
    SS.n = n;
    SS.xbar = mu;
    model = fit(model,'suffStat',SS,'prior',prior); % MAP
    mu = model.mu;
    Sigma = model.Sigma;
    
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

 
