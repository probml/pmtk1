function [models, bestModel, loglik, penLL] = selectPenLoglik(models, X, y, penalty)
Nm = length(models);
penLL = zeros(1, Nm);
loglik = zeros(1, Nm);
for m=1:Nm % for every model
  if isempty(y)
    models{m} = fit(models{m},  X);
    loglik(m) = sum(logprob(models{m}, X),1);
  else
    models{m} = fit(models{m}, X,  y);
    loglik(m) = sum(logprob(models{m}, X, y),1);
  end
  penLL(m) = loglik(m) - penalty*nparams(models{m}); %#ok
end
bestNdx = argmax(penLL);
bestModel = models{bestNdx};
end
