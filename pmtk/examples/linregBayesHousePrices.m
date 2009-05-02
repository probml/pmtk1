%% Bayesian linear regression with an informative prior
%#broken
% Example from Koop, 2003, p51, "Bayesian econometrics"

%#testPMTK
D = load('hprice.txt'); % 546 x   12
n=size(D,1); y=D(:,1); X=D(:,2:5);
%X=[ones(n,1) X];
% Prior parameters
w0 = [0, 10, 5000, 10000, 10000]';
S0 = diag([2.4, 6e-7, 0.15, 0.6, 0.6]);
s02 = 5000^2;
v0 = 5;
a0=v0/2;  b0 = v0*s02/2;
prior = MvnInvGammaDist('mu',w0,'Sigma',S0,'a',a0,'b',b0);
m = LinregConjugate('-wSigmaDist', prior);
m = fit(m, DataTable(X,y));
pw = marginal(m.wSigmaDist, 'mu'); % MVT
fprintf('posterior on coefficients\n');
for i=1:length(pw.mu)
  pwi = marginal(pw, i); % T
  mi = mean(pwi);
  si = sqrt(var(pwi));
  [l, u] = credibleInterval(pwi, 0.95);
  fprintf('%3d %10.2f +- %10.2f  (in %10.1f to %10.1f wp 0.95)\n', ...
    i, mi, si, l, u);
end
%xstar = [1 5000 2 2 1];
xstar = [5000 2 2 1];
[yhat,py] = predict(m, xstar);
fprintf('predicted price %5.3f +- %5.3f\n', mean(py), sqrt(var(py)));

assert(approxeq(yhat, 70468))