function [sigma_inv f] = covsel_mark(C,G,lambda);

nVars = length(C);

nonZero = G>0;
nonZero = nonZero(:);

funObj = @(x)sparsePrecisionObj(x,nVars,nonZero,C,lambda);
if(isempty(K_init))
  K_init = diag(rand(nVars,1));
end
options.TolX = 1e-16;
options.TolFun = 1e-16;
options.Method = 'newton0lbfgs';
options.MaxFunEvals = 1000000;
options.MaxIter = 100000;
options.Display = 'on';
%options.DerivativeCheck='on';
tic
[k,f] = minFunc(funObj,K_init(nonZero),options);
toc

K = zeros(nVars);
K(nonZero) = k;
sigma_inv = K;
