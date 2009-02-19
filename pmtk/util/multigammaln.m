% function [out] = multigammaln(a,d)
% Log of the multivariate gamma function
function [out] = multigammaln(a,d)

j = 1:d;
out = d*(d-1)/4*log(pi) + sum(gammaln(a + (1-j)/2));

end