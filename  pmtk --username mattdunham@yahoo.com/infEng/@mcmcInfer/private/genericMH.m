function [samples, naccept] = genericMH(target, proposal, xinit, Nburnin, Nsamples)
% Metropolis-Hastings algorithm
% 
% Inputs
% target returns the unnormalized log posterior, called as 'logp = target(x)'
% proposal is a fn, as [xprime, probOldToNew, probNewToOld] = proposal(x)' where x is a 1xd vector
% For symmetric proposals (Metropolis algo), set probOldToNew = probNewToOld = 1
% xinit is a 1xd vector specifying the initial state
% Nsamples - total number of samples to draw
%
% Outputs
% samples(s,:) is the s'th sample (of size d)
% naccept = number of accepted moves

d = length(xinit);
samples = zeros(Nsamples, d);
x = xinit(:)';
naccept = 0;
keep = 1;
logpOld = feval(target, x);
for iter=1:(Nsamples + Nburnin)
  [xprime, probOldToNew, probNewToOld] = feval(proposal, x);
  logpNew = feval(target, xprime);
  alpha = exp(logpNew - logpOld);
  % Hastings correction for asymmetric proposals
  alpha = alpha * (probNewToOld/probOldToNew);
  r = min(1, alpha);
  u = rand(1,1);
  if u < r
    x = xprime;
    naccept = naccept + 1;
    logpOld = logpNew;
  end
  if iter > Nburnin
    samples(keep,:) = x;
    keep = keep + 1;
  end
end
  
