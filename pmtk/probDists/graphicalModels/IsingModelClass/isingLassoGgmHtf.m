function [W] = isingLassoGgmHtf(X, lambda)
%  Find Ising structure using Gaussian graphical lasso 
% See  Banerjee, El Ghaoui, d'Aspremont, JMLR 2008 for details
% X(i,j) is case i, variable j, in {-1,+1}

if min(X(:))==0 % convert from 0,1 to -1,+1
  X = 2*X-1;
end

[n d] = size(X);
S = cov(X);
S = S - lambda*eye(d) + (1/3)*eye(d);
W = ggmLassoHtf(S, lambda);

end