function [muAgivenB, SigmaAgivenB] = gaussCondition(mu, Sigma, vnodes, vvals)
% Compute p(Xa|Xb=x(b)) = N(Xa|muAgivenB, SigmaAgivenB)

d = length(mu);
hnodes = setdiff(1:d, vnodes);
if isempty(hnodes)
  muAgivenB = []; SigmaAgivenB  = [];
  return;
end
b = vnodes; a = hnodes;
mu = mu(:); 
xb = vvals(:); % x(b)

muA = mu(a); muB = mu(b);
SAA = Sigma(a,a);
SAB = Sigma(a,b);
SBB = Sigma(b,b);
SBBinv = inv(SBB);
muAgivenB = mu(a) + SAB*SBBinv*(xb-mu(b));
SigmaAgivenB = SAA - SAB*SBBinv*SAB';
