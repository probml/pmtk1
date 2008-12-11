function [X] = gibbsSample(fullCond, xinit, Nsamples, Nburnin, thin)
%
% fullCond - fc{i}(x) returns a distribution which can be sampled
% xinit - a dx1 row vector
%
% OUTPUT
% X(s,:) = samples at step s

if nargin < 4, Nburnin = floor(0.1*Nsamples); end
if nargin < 5, thin = 1; end

keep = 1;
x = xinit;
S = (Nsamples*thin + Nburnin);
d = length(x);
X = zeros(Nsamples, d);
for iter=1:S
  for i=1:length(x)
    x(i) = sample(fullCond{i}(x));
  end
  if (iter > Nburnin) && (mod(iter, thin)==0)
    X(keep,:) = x; keep = keep + 1;
  end
end
end
