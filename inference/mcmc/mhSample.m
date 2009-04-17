function [X, acceptRatio] = mhSample(varargin)
% Metropolis Hastings algorithm
% symmetric - true or false
% target - logp = target(x)
% proposal - if symmetric=true, xprime = proposal(x)
%          - if symmetric = false, [xprime, probOldToNew, probNewToOld] =proposal(x)
% Nsamples
% Nburnin
% thin
% xinit - a dx1 row vector
%
% OUTPUT
% X(s,:) = samples at step s

[symmetric, target, proposal, Nsamples, Nburnin, thin, xinit] = ...
  process_options(varargin, 'symmetric', true, 'target', [], ...
  'proposal', [], 'Nsamples', 1000, 'Nburnin', 100, 'thin', 1, 'xinit', []);

keep = 1;
x = xinit;
logpx = target(x);
S = (Nsamples*thin + Nburnin);
d = length(x);
X = zeros(Nsamples, d);
u = rand(S,1); % move outside main loop to speedup MH
naccept = 0;
for iter=1:S
  [x, accept, logpx] = mhUpdate(x, logpx, u(iter), proposal, target, symmetric); 
  if (iter > Nburnin) && (mod(iter, thin)==0)
    X(keep,:) = x; keep = keep + 1;
    naccept = naccept + accept;
  end
end
acceptRatio = naccept/(keep-1);
end


function [xnew, accept, logpNew] = mhUpdate(x, logpOld, u, proposal, target, symmetric)
if symmetric
  [xprime] = proposal(x);
  probOldToNew = 1; probNewToOld = 1;
else
  [xprime, probOldToNew, probNewToOld] = proposal(x);
end
logpNew = target(xprime); 
alpha = exp(logpNew - logpOld);
alpha = alpha * (probNewToOld/probOldToNew);  % Hastings correction for asymmetric proposals
r = min(1, alpha);
%u = rand(1,1);
if u < r
  xnew = xprime;
  accept = 1;
else
  accept = 0;
  xnew = x;
  logpNew = logpOld;
end
end
    